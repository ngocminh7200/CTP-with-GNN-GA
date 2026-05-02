import pandas as pd
import numpy as np
import torch
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


def create_dataset(instance_file, room_file, poor_schedule_file, target_file, extra_degrade_ratios=None,
                   extra_per_ratio=0):
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
    size = target_df['sample_id'].max()
    for this_instance in target_df['sample_id'].unique():
    # for this_instance in range(5):
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
        room_to_idx = {r: i for i, r in enumerate(
            instance.rooms['Host Key'].drop_duplicates())}

        target_ts = torch.zeros(len(instance.course_list), dtype=torch.long)
        target_room = torch.zeros(len(instance.course_list), dtype=torch.long)
        for _, row in target_sub_df.iterrows():
            ci = course_to_idx.get(row['Index'])
            if ci is not None:
                if row['timeslot'] in timeslot_to_idx:
                    target_ts[ci] = timeslot_to_idx[row['timeslot']]
                if row['Room'] in room_to_idx:
                    target_room[ci] = room_to_idx[row['Room']]

        # 1. Original poor schedule graph
        g = Graph()
        g.ds = instance
        g.sched = poor_sub_df
        data = g.build_graph()
        data['course'].y_ts = target_ts.clone()
        data['course'].y_room = target_room.clone()
        data.problem_size = len(instance.course_list)
        data.num_rooms = len(room_to_idx)
        dataset.append(data)

        # 2. Generate additional degraded versions from the optimized schedule
        if extra_degrade_ratios and extra_per_ratio > 0:
            for ratio in extra_degrade_ratios:
                for _ in range(extra_per_ratio):
                    degraded_df = degrade_schedule(target_sub_df, degrade_ratio=ratio)
                    g_deg = Graph()
                    g_deg.ds = instance
                    g_deg.sched = degraded_df
                    data_deg = g_deg.build_graph()
                    data_deg['course'].y_ts = target_ts.clone()
                    data_deg['course'].y_room = target_room.clone()
                    data_deg.problem_size = len(instance.course_list)
                    data_deg.num_rooms = len(room_to_idx)
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cases = [[50, 10], [100, 20], [150, 25], [200, 30], [300, 37], [400, 50]]

    # --- Build dataset ---
    # dataset = create_dataset(
    #     "Sample/courses_sampling_400.csv",
    #     "Sample/rooms_sampling_50.csv",
    #     "Sample/poor_schedule_generating_400.csv",
    #     "Sample/optimized_schedule_400.csv"
    # )
    for course, room in cases:
        dataset = create_dataset(
            'Sample/courses_sampling_' + str(course) + '.csv',
            'Sample/rooms_sampling_' + str(room) + '.csv',
            'Sample/poor_schedule_generating_' + str(course) + '.csv',
            'Sample/optimized_schedule_' + str(course) + '.csv'
        )
        torch.save(dataset, 'TrainingData/dataset_' + str(course) + '.pt')
