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
    from the batched edge_index_dict. This works correctly with PyG batching
    because edge indices are automatically remapped.

    Each course has exactly one edge to its professor, room, program, and
    timeslot, so we can recover the mapping from the forward edge indices.
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
        course_indices = edge_index[0]  # source: course nodes
        target_indices = edge_index[1]  # dest: professor/room/program/timeslot

        # Each course has exactly one such edge. Sort by course index
        # to get a tensor aligned with course node ordering.
        mapping = torch.zeros(num_courses, dtype=torch.long,
                              device=course_indices.device)
        mapping[course_indices] = target_indices
        mappings[key] = mapping

    return mappings

# def train(model):
#     model.train()
#
#     for data in train_loader:  # Iterate in batches over the training dataset.
#         out = model(
#             data.x, data.edge_index, data.batch
#         )  # Perform a single forward pass.
#         loss = criterion(out, data.y)  # Compute the loss.
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_courses = 0

    for data in loader:
        data = data.to(device)
        num_courses = data['course'].x.size(0)

        # Extract mapping tensors from batched edge indices
        mappings = extract_mappings_from_edges(data.edge_index_dict, num_courses)

        optimizer.zero_grad()

        logits = model(
            data.x_dict,
            data.edge_index_dict,
            mappings['professor'],
            mappings['room'],
            mappings['program'],
            mappings['timeslot'],
        )  # [total_courses_in_batch, num_timeslots]

        targets = data['course'].y  # [total_courses_in_batch]
        loss = F.cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_courses
        total_courses += num_courses

    return total_loss / total_courses


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate using the paper's validation metric:
    - Primary: fraction of schedules that are optimal or feasible
    - Also returns average cross-entropy loss and per-course accuracy
    """
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
        )

        # Timeslot loss + accuracy
        ts_targets = data['course'].y_ts
        ts_loss = F.cross_entropy(ts_logits, ts_targets)
        ts_preds = ts_logits.argmax(dim=-1)
        ts_correct = (ts_preds == ts_targets)

        # Room loss + accuracy (using local targets)
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
    """
    Run the GNN on a graph and convert predictions to a schedule DataFrame.
    """
    model.eval()
    data = data.to(device)
    num_courses = data['course'].x.size(0)

    # Extract mapping tensors from edge indices
    mappings = extract_mappings_from_edges(data.edge_index_dict, num_courses)

    # Run model
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

    # Build reverse mappings: index -> actual value
    idx_to_timeslot = {i: t for i, t in enumerate(list(instance.timeslots))}
    idx_to_room = {i: r for i, r in enumerate(
        instance.rooms['Host Key'].drop_duplicates().tolist())}
    idx_to_course = {i: c for i, c in enumerate(instance.course_list)}

    # Build schedule DataFrame
    rows = []
    for c in range(num_courses):
        rows.append({
            'Index': idx_to_course[c],
            'timeslot': idx_to_timeslot[ts_preds[c]],
            'Room': idx_to_room[room_preds[c]],
        })

    predicted_schedule = pd.DataFrame(rows)
    return predicted_schedule



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cases = [50, 100, 150, 200, 300, 400]

    dataset = []

    for case in cases:
        dataset += torch.load('TrainingData/dataset_' + str(case) + '.pt', weights_only=False)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    print(f"Dataset: {n} graphs (train={len(train_data)}, "
          f"val={len(val_data)}, test={len(test_data)})")

    # Inspect one batch
    for data in train_loader:
        print(f"\nSample batch: {data}")
        print(f"Course nodes: {data['course'].x.shape}")
        print(f"Edge types: {list(data.edge_index_dict.keys())}")
        break

    for i in range(3):
        x = dataset[i]['course'].x
        print(f"Instance {i}: shape={x.shape}")

    for i in range(5):
        data = dataset[i]
        x = data['course'].x
        print(f"Instance {i}: {x.shape[0]} courses | "
              f"h2 conflicts: {x[:, 0].sum():.0f} | "
              f"h3 conflicts: {x[:, 1].sum():.0f} | "
              f"h4 violations: {x[:, 2].sum():.0f}")

        print(f"Instance {i}: {x.shape[0]} courses | "
              f"h2 prof conflicts: {x[:, 0].sum():.0f} | "
              f"h3 room conflicts: {x[:, 1].sum():.0f} | "
              f"h4 capacity violations: {x[:, 2].sum():.0f} | "
              f"h4 overflow mean: {x[:, 3].mean():.3f}")

        y = data['course'].y

        # Get current timeslot assignment from edges
        edge = data['course', 'when', 'timeslot'].edge_index
        current_ts = torch.zeros(y.shape[0], dtype=torch.long)
        current_ts[edge[0]] = edge[1]

        # How many targets differ from input? (should be most of them if degraded)
        changed = (y != current_ts).sum().item()
        same = (y == current_ts).sum().item()

        print(f"Instance {i}: {y.shape[0]} courses | "
              f"target range [{y.min()}-{y.max()}] | "
              f"same as input: {same} | different: {changed}")


    for i in range(5):
        x = dataset[i]['course'].x
        room_avail = x[:, 6:31]
        avg_available = room_avail.mean(dim=1).mean().item()
        zero_slots = (room_avail == 0).float().mean().item()
        print(f"Instance {i}: avg room availability: {avg_available:.3f} | "
              f"zero-availability slots: {zero_slots:.1%}")

    # --- Model ---
    model = GNN(
        num_timeslots=25,
        hidden_channels=64,
        nb_layers_conv=4,
        nb_layers_mlp=4,
        nb_heads=8,
        dropout_conv=0.0,
        dropout_mlp=0.0,
    ).to(device)

    # AdamW with weight_decay=0.0001, lr=0.001 (paper's best config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    print(f"\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Training with early stopping (patience=30 per paper) ---
    patience = 30
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    nb_max_epochs = 200

    print("\n--- Training ---")
    for epoch in range(1, nb_max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epochs_without_improvement == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Patience: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # --- Final evaluation on test set ---
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n--- Test Results ---")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # --- Save model ---
    torch.save(model.state_dict(), "timetabling_gnn_trained.pt")
    print("\nModel saved to timetabling_gnn_trained.pt")