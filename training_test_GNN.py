import os
import copy

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from GNN import GNN
from Graph import Graph
from Instance import Instance


def extract_mappings_from_edges(edge_index_dict, num_courses):
    """
    Extract course -> {professor, room, program, timeslot} mapping tensors
    from the batched edge_index_dict.

    Each course is expected to have exactly one edge of each type. If any
    course is missing an edge of a given type, this raises an AssertionError
    rather than silently mapping it to index 0.
    """
    mappings = {}
    edge_type_to_key = {
        ('course', 'taught_by', 'professor'): 'professor',
        ('course', 'take_place_in', 'Room'): 'room',
        ('course', 'belong_to', 'program'): 'program',
        ('course', 'when', 'timeslot'): 'timeslot',
    }

    for edge_type, key in edge_type_to_key.items():
        edge_index = edge_index_dict[edge_type]
        course_indices = edge_index[0]   # source: course nodes
        target_indices = edge_index[1]   # dest: prof/room/program/timeslot

        # CHANGED: -1 sentinel + assertion catches missing edges instead of
        # silently mapping to index 0.
        mapping = torch.full(
            (num_courses,), -1,
            dtype=torch.long, device=course_indices.device,
        )
        mapping[course_indices] = target_indices
        if (mapping < 0).any():
            missing = (mapping < 0).nonzero(as_tuple=True)[0]
            raise AssertionError(
                f"{missing.numel()} courses are missing a {key} edge. "
                f"Indices (first 10): {missing[:10].tolist()}"
            )

        mappings[key] = mapping

    return mappings


# CHANGED: vectorised. Replaces the per-element Python loop that synchronised
# GPU->CPU on every iteration.
def build_room_local_targets(room_batch, course_batch, y_room_global):
    """
    Convert global room target indices into local (per-graph) indices.

    PyG's batching gives room nodes contiguous global indices, ordered by
    graph. Within a single graph, room order matches the order in which
    rooms were added to data['Room'].x. The "local" index of a room is
    therefore its position among rooms of the same graph.

    Vectorised implementation: for each room node we count how many earlier
    nodes belong to the same graph using a one-hot cumulative sum.
    """
    num_graphs = int(room_batch.max().item()) + 1

    # one-hot per room node, of shape [R_total, num_graphs]
    is_in_graph = F.one_hot(room_batch, num_classes=num_graphs)
    # cumulative count of rooms per graph BEFORE the current row -> -1 for inclusive
    cumulative = is_in_graph.cumsum(dim=0) - 1   # [R_total, num_graphs]
    # gather along the room's own graph column
    global_to_local = cumulative.gather(1, room_batch.unsqueeze(1)).squeeze(1)

    return global_to_local[y_room_global]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_courses = 0

    for data in loader:
        data = data.to(device)
        num_courses = data['course'].x.size(0)

        mappings = extract_mappings_from_edges(data.edge_index_dict, num_courses)

        course_batch = data['course'].batch if hasattr(data['course'], 'batch') else None
        room_batch = data['Room'].batch if hasattr(data['Room'], 'batch') else None

        optimizer.zero_grad()

        ts_logits, room_logits = model(
            data.x_dict,
            data.edge_index_dict,
            mappings['professor'],
            mappings['room'],
            mappings['program'],
            mappings['timeslot'],
            course_batch=course_batch,
            room_batch=room_batch,
        )

        ts_targets = data['course'].y_ts
        ts_loss = F.cross_entropy(ts_logits, ts_targets)

        if room_batch is not None and course_batch is not None:
            room_targets_local = build_room_local_targets(
                room_batch, course_batch, data['course'].y_room
            )
        else:
            room_targets_local = data['course'].y_room

        room_loss = F.cross_entropy(room_logits, room_targets_local)
        loss = ts_loss + room_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * num_courses
        total_courses += num_courses

    return total_loss / total_courses


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_ts_correct = 0
    total_room_correct = 0
    total_both_correct = 0
    total_courses = 0

    for data in loader:
        data = data.to(device)
        num_courses = data['course'].x.size(0)

        mappings = extract_mappings_from_edges(data.edge_index_dict, num_courses)

        course_batch = data['course'].batch if hasattr(data['course'], 'batch') else None
        room_batch = data['Room'].batch if hasattr(data['Room'], 'batch') else None

        ts_logits, room_logits = model(
            data.x_dict,
            data.edge_index_dict,
            mappings['professor'],
            mappings['room'],
            mappings['program'],
            mappings['timeslot'],
            course_batch=course_batch,
            room_batch=room_batch,
        )

        ts_targets = data['course'].y_ts
        ts_loss = F.cross_entropy(ts_logits, ts_targets)
        ts_preds = ts_logits.argmax(dim=-1)
        ts_correct = (ts_preds == ts_targets)

        if room_batch is not None and course_batch is not None:
            room_targets_local = build_room_local_targets(
                room_batch, course_batch, data['course'].y_room
            )
        else:
            room_targets_local = data['course'].y_room

        room_loss = F.cross_entropy(room_logits, room_targets_local)
        room_preds = room_logits.argmax(dim=-1)
        room_correct = (room_preds == room_targets_local)

        loss = ts_loss + room_loss

        total_loss += loss.item() * num_courses
        total_ts_correct += ts_correct.sum().item()
        total_room_correct += room_correct.sum().item()
        total_both_correct += (ts_correct & room_correct).sum().item()
        total_courses += num_courses

    avg_loss = total_loss / total_courses
    ts_acc = total_ts_correct / total_courses
    room_acc = total_room_correct / total_courses
    both_acc = total_both_correct / total_courses
    return avg_loss, ts_acc, room_acc, both_acc


def predict_schedule(model, data, instance, device='cpu'):
    """Run the GNN on a single graph and return a schedule DataFrame."""
    model.eval()
    data = data.to(device)
    num_courses = data['course'].x.size(0)

    mappings = extract_mappings_from_edges(data.edge_index_dict, num_courses)

    with torch.no_grad():
        ts_logits, room_logits = model(
            data.x_dict,
            data.edge_index_dict,
            mappings['professor'],
            mappings['room'],
            mappings['program'],
            mappings['timeslot'],
        )

    ts_preds = ts_logits.argmax(dim=-1).cpu().numpy()
    room_preds = room_logits.argmax(dim=-1).cpu().numpy()

    idx_to_timeslot = {i: t for i, t in enumerate(list(instance.timeslots))}
    idx_to_room = {i: r for i, r in enumerate(
        instance.rooms['Host Key'].drop_duplicates().tolist())}
    idx_to_course = {i: c for i, c in enumerate(instance.course_list)}

    rows = []
    for c in range(num_courses):
        rows.append({
            'Index': idx_to_course[c],
            'timeslot': idx_to_timeslot[ts_preds[c]],
            'Room': idx_to_room[room_preds[c]],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cases = [50, 100, 150, 200, 300, 400,800,1000]
    #cases = [800]
    dataset = []
    for case in cases:
        path = f'TrainingData/dataset_{case}.pt'
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training dataset {path} not found. Generate it first."
            )
        dataset += torch.load(path, weights_only=False)

    # CHANGED: keep the dataset on CPU. The DataLoader will move each batch
    # to `device` at consumption time (see train_one_epoch / evaluate).
    # Mutating dataset[0] in place — as the previous version did with
    # `g = g.to(device)` — corrupts batching because dataset[0] ends up on
    # CUDA while dataset[1..7] remain on CPU.

    # Inspect a single graph (without moving it).
    g_inspect = dataset[0]
    print(f"\n--- Single graph (inspection only) ---")
    print(f"  Node types: {list(g_inspect.x_dict.keys())}")
    for ntype, x in g_inspect.x_dict.items():
        print(f"  {ntype:>12}: {x.shape[0]:>4} nodes × {x.shape[1]} features")
    print(f"  Edge types: {list(g_inspect.edge_index_dict.keys())}")
    print(f"  y_ts  range: [{g_inspect['course'].y_ts.min()}, "
          f"{g_inspect['course'].y_ts.max()}]")
    print(f"  y_room range: [{g_inspect['course'].y_room.min()}, "
          f"{g_inspect['course'].y_room.max()}]")

    # Build model with auto-detected dims.
    feature_dims = {
        'course_feature_dim': g_inspect['course'].x.size(1),
        'professor_feature_dim': g_inspect['professor'].x.size(1),
        'room_feature_dim': g_inspect['Room'].x.size(1),
        'program_feature_dim': g_inspect['program'].x.size(1),
        'timeslot_feature_dim': g_inspect['timeslot'].x.size(1),
    }
    print(f"\nFeature dims: {feature_dims}")

    model = GNN(
        num_timeslots=g_inspect['timeslot'].x.size(0),  # CHANGED: derive from data
        hidden_channels=64,
        nb_layers_conv=4,
        nb_layers_mlp=4,
        nb_heads=8,
        **feature_dims,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # ---- Single graph forward pass ----
    # CHANGED: deep-copy so the dataset entry stays on CPU.
    print("--- Single graph forward pass ---")
    g = copy.copy(g_inspect).to(device)
    num_courses = g['course'].x.size(0)
    mappings = extract_mappings_from_edges(g.edge_index_dict, num_courses)

    model.eval()
    with torch.no_grad():
        ts_logits, room_logits = model(
            g.x_dict, g.edge_index_dict,
            mappings['professor'], mappings['room'],
            mappings['program'], mappings['timeslot'],
        )
    expected_ts = g['timeslot'].x.size(0)
    expected_rooms = g['Room'].x.size(0)
    print(f"  ts_logits:   {ts_logits.shape}  (expect [{num_courses}, {expected_ts}])")
    print(f"  room_logits: {room_logits.shape}  (expect [{num_courses}, {expected_rooms}])")

    ts_preds = ts_logits.argmax(dim=-1)
    room_preds = room_logits.argmax(dim=-1)
    assert ts_logits.shape == (num_courses, expected_ts), "ts_logits shape mismatch!"
    assert room_logits.shape == (num_courses, expected_rooms), "room_logits shape mismatch!"
    assert room_preds.max() < expected_rooms, "Room prediction out of range!"
    print("  ✓ Shapes and ranges correct")

    # Free the GPU copy now that we're done with it.
    del g

    # ---- Batched forward pass ----
    print("\n--- Batched forward pass (batch_size=4) ---")
    loader = DataLoader(dataset[:min(8, len(dataset))], batch_size=4, shuffle=False)
    batch = next(iter(loader)).to(device)   # safe now: dataset entries stayed on CPU
    num_courses_batch = batch['course'].x.size(0)
    mappings_b = extract_mappings_from_edges(batch.edge_index_dict, num_courses_batch)

    course_batch = batch['course'].batch
    room_batch = batch['Room'].batch

    with torch.no_grad():
        ts_logits_b, room_logits_b = model(
            batch.x_dict, batch.edge_index_dict,
            mappings_b['professor'], mappings_b['room'],
            mappings_b['program'], mappings_b['timeslot'],
            course_batch=course_batch,
            room_batch=room_batch,
        )
    print(f"  ts_logits:   {ts_logits_b.shape}")
    print(f"  room_logits: {room_logits_b.shape}")

    rooms_per_graph = torch.zeros(course_batch.max() + 1, dtype=torch.long, device=device)
    rooms_per_graph.scatter_add_(0, room_batch, torch.ones_like(room_batch))
    R_max = rooms_per_graph.max().item()
    assert room_logits_b.shape[1] == R_max, (
        f"Room logits width {room_logits_b.shape[1]} != R_max {R_max}"
    )
    print(f"  R_max in batch: {R_max}")

    # ---- One training step ----
    print("\n--- Training step ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    ts_logits_b, room_logits_b = model(
        batch.x_dict, batch.edge_index_dict,
        mappings_b['professor'], mappings_b['room'],
        mappings_b['program'], mappings_b['timeslot'],
        course_batch=course_batch,
        room_batch=room_batch,
    )

    ts_loss = F.cross_entropy(ts_logits_b, batch['course'].y_ts)
    room_targets_local = build_room_local_targets(
        room_batch, course_batch, batch['course'].y_room
    )
    room_loss = F.cross_entropy(room_logits_b, room_targets_local)
    loss = ts_loss + room_loss

    loss.backward()

    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if not torch.isfinite(p.grad).all():
                print(f"  WARNING: non-finite gradient in {name}")

    optimizer.step()

    print(f"  ts_loss:   {ts_loss.item():.4f}  "
          f"(random baseline ~{torch.log(torch.tensor(float(expected_ts))).item():.2f})")
    print(f"  room_loss: {room_loss.item():.4f}  "
          f"(random baseline ~{torch.log(torch.tensor(float(R_max))).item():.2f})")
    print(f"  total:     {loss.item():.4f}")
    print(f"  grad norms: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, "
          f"mean={sum(grad_norms) / len(grad_norms):.4f}")
    print(f"  params with gradients: {len(grad_norms)}/"
          f"{sum(1 for _ in model.parameters())}")
    print("  ✓ Backward pass and optimizer step successful")

    # ---- Quick overfit test ----
    print("\n--- Overfit test (50 steps, 1 batch) ---")
    print(f"  {'Step':>4} | {'Loss':>8} | {'TS Acc':>7} | {'Rm Acc':>7} | {'Both':>7}")
    print("  " + "-" * 48)
    losses = []
    for step in range(1, 51):
        optimizer.zero_grad()
        ts_out, room_out = model(
            batch.x_dict, batch.edge_index_dict,
            mappings_b['professor'], mappings_b['room'],
            mappings_b['program'], mappings_b['timeslot'],
            course_batch=course_batch,
            room_batch=room_batch,
        )
        # CHANGED: recompute room_targets_local each step. Targets don't
        # change for a fixed batch, but recomputing makes the loop self-
        # contained and harder to break later.
        room_targets_local = build_room_local_targets(
            room_batch, course_batch, batch['course'].y_room
        )
        l = F.cross_entropy(ts_out, batch['course'].y_ts) + \
            F.cross_entropy(room_out, room_targets_local)
        l.backward()
        optimizer.step()
        losses.append(l.item())

        ts_correct = (ts_out.argmax(dim=-1) == batch['course'].y_ts)
        room_correct = (room_out.argmax(dim=-1) == room_targets_local)
        ts_acc = ts_correct.float().mean().item()
        room_acc = room_correct.float().mean().item()
        both_acc = (ts_correct & room_correct).float().mean().item()

        if step <= 5 or step % 10 == 0:
            print(f"  {step:4d} | {l.item():8.4f} | {ts_acc:6.2%} | {room_acc:6.2%} | {both_acc:6.2%}")