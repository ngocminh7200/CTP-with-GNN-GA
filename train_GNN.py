import pandas as pd
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from GNN import GNN
from Graph import Graph
from Instance import Instance


def degrade_schedule(schedule_df, degrade_ratio=0.2):
    """
    Degrade an optimized schedule by randomly reassigning timeslots and rooms
    for a fraction of courses. Following the paper's Section 4.2 approach.

    Args:
        schedule_df: DataFrame with an optimized schedule
        degrade_ratio: Fraction of courses to randomly reassign (0.0 to 1.0)

    Returns:
        DataFrame with degraded assignments (same structure, different timeslots/rooms)
    """
    degraded = schedule_df.copy()
    num_courses = len(degraded)
    num_to_swap = max(1, int(num_courses * degrade_ratio))

    swap_indices = np.random.choice(num_courses, size=num_to_swap, replace=False)

    valid_timeslots = degraded['timeslot'].unique().tolist()
    valid_rooms = degraded['Room'].unique().tolist()

    for idx in swap_indices:
        # Swap timeslot
        current_ts = degraded.iloc[idx]['timeslot']
        other_ts = [t for t in valid_timeslots if t != current_ts]
        if other_ts:
            degraded.iloc[idx, degraded.columns.get_loc('timeslot')] = np.random.choice(other_ts)

        # Optionally swap room too (introduces capacity + room conflict violations)
        if np.random.random() < 0.5:
            current_room = degraded.iloc[idx]['Room']
            other_rooms = [r for r in valid_rooms if r != current_room]
            if other_rooms:
                degraded.iloc[idx, degraded.columns.get_loc('Room')] = np.random.choice(other_rooms)

    return degraded

def create_dataset(instance_file, room_file, poor_schedule_file, target_file):
    dataset = []

    course_df = pd.read_csv(instance_file)
    room_df = pd.read_csv(room_file)
    poor_df = pd.read_csv(poor_schedule_file)
    target_df = pd.read_csv(target_file)
    course_df['P-Index'] = course_df['P-Index'].fillna(0)
    course_df['P-Index'] = course_df['P-Index'].astype(float).astype(int).astype(str)
    course_df['Index'] = course_df['Index'].astype(float).astype(int).astype(str)

    target_df.rename(columns={'course index': 'Index'}, inplace=True)
    target_df.rename(columns={'room': 'Host Key'}, inplace=True)
    target_df['Index'] = target_df['Index'].astype(float).astype(int).astype(str)

    target_df = pd.merge(target_df, course_df, on=['Index', 'sample_id'])
    target_df = pd.merge(target_df, room_df, on=['Host Key', 'sample_id'])

    target_df['P-Index'] = target_df['P-Index'].astype(float).astype(int).astype(str)
    target_df.rename(columns={'Host Key': 'Room'}, inplace=True)

    poor_df.rename(columns={'course index': 'Index'}, inplace=True)
    poor_df.rename(columns={'classroom index': 'R-Index'}, inplace=True)
    # poor_df.rename(columns={'professor': 'P-Index'}, inplace=True)
    poor_df['Index'] = poor_df['Index'].astype(float).astype(int).astype(str)

    poor_df = pd.merge(poor_df, course_df, on=['Index', 'sample_id', 'dl'])
    poor_df = pd.merge(poor_df, room_df, on=['R-Index', 'sample_id'])

    poor_df.rename(columns={'Host Key': 'Room'}, inplace=True)
    # poor_df.rename(columns={'Host Key': 'Room'}, inplace=True)

    # professor_df = target_df[['P-Index', 'Index', 'sample_id']]
    # poor_professor_df = poor_df[['P-Index', 'Index', 'sample_id']]
    print(poor_df.columns)

    # size = course_df['sample_id'].max() + 1
    # size = 10

    # for this_instance in range(size):
    #     course_sub_df = course_df[course_df['sample_id'] == this_instance]
    #     room_sub_df = room_df[room_df['sample_id'] == this_instance]
    #     poor_sub_df = poor_df[poor_df['sample_id'] == this_instance]
    #     target_sub_df = target_df[target_df['sample_id'] == this_instance]
    #
    #     if len(poor_sub_df) == 0 or len(target_sub_df) == 0:
    #         continue
    #
    #     # Build instance from poor schedule
    #     poor_professor_df = poor_sub_df[['P-Index', 'Index', 'sample_id']]
    #     instance = Instance(course_sub_df, room_sub_df, 25, poor_professor_df)
    #
    #     # --- Build graph from POOR schedule (input features show violations) ---
    #     g = Graph()
    #     g.ds = instance
    #     g.sched = poor_sub_df
    #     data = g.build_graph()
    #
    #     # --- Extract target timeslots from OPTIMIZED schedule ---
    #     timeslot_to_idx = {t: i for i, t in enumerate(list(instance.timeslots))}
    #     course_to_idx = {c: i for i, c in enumerate(instance.course_list)}
    #
    #     target_ts = torch.zeros(len(instance.course_list), dtype=torch.long)
    #     for _, row in target_sub_df.iterrows():
    #         ci = course_to_idx.get(row['Index'])
    #         if ci is not None and row['timeslot'] in timeslot_to_idx:
    #             target_ts[ci] = timeslot_to_idx[row['timeslot']]
    #
    #     # Attach optimized timeslots as the training target
    #     data['course'].y = target_ts
    #
    #     dataset.append(data)
    size = len(target_df['sample_id'].unique())
    for this_instance in target_df['sample_id'].unique():
        print(f"  Processing {this_instance + 1}/{size} instance ")
        course_sub_df = course_df[course_df['sample_id'] == this_instance]
        room_sub_df = room_df[room_df['sample_id'] == this_instance]
        poor_sub_df = poor_df[poor_df['sample_id'] == this_instance]
        target_sub_df = target_df[target_df['sample_id'] == this_instance]

        if len(poor_sub_df) == 0 or len(target_sub_df) == 0:
            continue

        poor_professor_df = poor_sub_df[['P-Index', 'Index', 'sample_id']]
        instance = Instance(course_sub_df, room_sub_df, 25, poor_professor_df)

        timeslot_to_idx = {t: i for i, t in enumerate(list(instance.timeslots))}
        course_to_idx = {c: i for i, c in enumerate(instance.course_list)}

        target_ts = torch.zeros(len(instance.course_list), dtype=torch.long)
        for _, row in target_sub_df.iterrows():
            ci = course_to_idx.get(row['Index'])
            if ci is not None and row['timeslot'] in timeslot_to_idx:
                target_ts[ci] = timeslot_to_idx[row['timeslot']]

        # 1. Append the original poor schedule
        g = Graph()
        g.ds = instance
        g.sched = poor_sub_df
        data = g.build_graph()
        data['course'].y = target_ts.clone()
        dataset.append(data)

        # 2. Generate additional degraded versions from the optimized schedule
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            for _ in range(50 // 4):
                degraded_sub_df = degrade_schedule(target_sub_df, degrade_ratio=ratio)
                g_deg = Graph()
                g_deg.ds = instance
                g_deg.sched = degraded_sub_df
                data_deg = g_deg.build_graph()
                data_deg['course'].y = target_ts.clone()
                dataset.append(data_deg)

    print(f"Created {len(dataset)} training samples from {size} instances")
    return dataset

    # for this_instance in range(size):
    #     g_target = Graph()
    #
    #     course_sub_df = course_df[course_df['sample_id'] == this_instance]
    #     room_sub_df = room_df[room_df['sample_id'] == this_instance]
    #     professor_sub_df = professor_df[professor_df['sample_id'] == this_instance]
    #     target_sub_df = target_df[target_df['sample_id'] == this_instance]
    #
    #     g_target.ds = Instance(course_sub_df, room_sub_df, 25, professor_sub_df)
    #     g_target.sched = target_sub_df
    #     data = g_target.build_graph()
    #     dataset.append(data)
    #
    #     g_poor = Graph()
    #
    #     course_sub_df = course_df[course_df['sample_id'] == this_instance]
    #     room_sub_df = room_df[room_df['sample_id'] == this_instance]
    #     professor_sub_df = poor_professor_df[poor_professor_df['sample_id'] == this_instance]
    #     poor_sub_df = poor_df[poor_df['sample_id'] == this_instance]
    #
    #     g_poor.ds = Instance(course_sub_df, room_sub_df, 25, professor_sub_df)
    #     g_poor.sched = poor_sub_df
    #     data = g_poor.build_graph()
    #     dataset.append(data)
    #
    # return dataset


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
    total_correct = 0
    total_courses = 0

    for data in loader:
        data = data.to(device)
        num_courses = data['course'].x.size(0)

        mappings = extract_mappings_from_edges(data.edge_index_dict, num_courses)

        logits = model(
            data.x_dict,
            data.edge_index_dict,
            mappings['professor'],
            mappings['room'],
            mappings['program'],
            mappings['timeslot'],
        )

        targets = data['course'].y
        loss = F.cross_entropy(logits, targets)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * num_courses
        total_correct += (preds == targets).sum().item()
        total_courses += num_courses

    avg_loss = total_loss / total_courses
    accuracy = total_correct / total_courses
    return avg_loss, accuracy


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Build dataset ---
    dataset = create_dataset(
        "Sample/courses_sampling_400.csv",
        "Sample/rooms_sampling_50.csv",
        "Sample/poor_schedule_generating_400.csv",
        "Sample/optimized_schedule_400.csv"
    )

    # Quick sanity check
    for i in range(min(5, len(dataset))):
        y = dataset[i]['course'].y
        print(f"Instance {i}: target min={y.min().item()}, max={y.max().item()}, "
              f"unique={y.unique().shape[0]}, zeros={(y == 0).sum().item()}/{y.shape[0]}")

    # --- Split: 80/10/10 as in the paper ---
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

    for i in range(5):
        data = dataset[i]
        x = data['course'].x
        print(f"Instance {i}: {x.shape[0]} courses | "
              f"h2 prof conflicts: {x[:, 0].sum():.0f} | "
              f"h3 room conflicts: {x[:, 1].sum():.0f} | "
              f"h4 capacity violations: {x[:, 2].sum():.0f} | "
              f"h4 overflow mean: {x[:, 3].mean():.3f}")

    # for i in range(5):
    #     x = dataset[i]['course'].x
    #     y = dataset[i]['course'].y
    #     prof_avail = x[:, 6:31]
    #
    #     target_available = 0
    #     for c in range(y.shape[0]):
    #         target_ts = y[c].item()
    #         if prof_avail[c, target_ts] == 1.0:
    #             target_available += 1
    #
    #     pct = target_available / y.shape[0] * 100
    #     avg_feasible = prof_avail.sum(dim=1).mean().item()
    #     print(f"Instance {i}: targets in available slots: {pct:.1f}% | "
    #           f"avg feasible timeslots per course: {avg_feasible:.1f}/25")

    for i in range(5):
        data = dataset[i]
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