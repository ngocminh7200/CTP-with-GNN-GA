import pandas as pd

timetable_by_course_code = pd.read_excel('Data/Timetable_by_course_code_2025_26.xlsx')
timetable_by_course_code.rename(columns={'Module.Host Key': 'Module'}, inplace=True)
timetable_by_course_code['Code'] = timetable_by_course_code['Description'].str.split(' ').str[0]
timetable_by_course_code['Start'] = pd.to_timedelta(timetable_by_course_code['Start'] + ':00')
timetable_by_course_code['End'] = pd.to_timedelta(timetable_by_course_code['End'] + ':00')
timetable_by_course_code['Duration'] = timetable_by_course_code['End'] - timetable_by_course_code['Start']
timetable_by_course_code['AT'] =  timetable_by_course_code['AT Weeks'].notna() & (timetable_by_course_code['AT Weeks'] != '')
timetable_by_course_code['WT'] =  timetable_by_course_code['WT Weeks'].notna() & (timetable_by_course_code['WT Weeks'] != '')
timetable_by_course_code['AT_week_count'] = timetable_by_course_code['AT Weeks'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0)
timetable_by_course_code['WT_week_count'] = timetable_by_course_code['WT Weeks'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0)

teaching_rooms = pd.read_excel('Data/Teaching_rooms_2025_26.xlsx', header=2)

groups = pd.read_excel('Data/Group Sizes.xlsx')
groups.rename(columns={"Group No.": "Group"}, inplace=True)

course_master_data = timetable_by_course_code[['Code', 'Type', 'Group', 'Department Name', 'Module', 'Description', 'Duration', 'AT', 'AT Weeks', 'WT', 'WT Weeks', 'Staff']].copy()
course_master_data['Group'] = course_master_data['Group'].astype(str)
course_master_data.loc[course_master_data['Type'] == 'LECTURE', 'Group'] = 'L'
course_master_data.loc[course_master_data['Type'] == 'WORKSHOP', 'Group'] = 'W'
course_master_data = pd.merge(course_master_data, groups[['Code', 'Group', 'NUMSTU']].copy(), on=['Code', 'Group'])
course_master_data['Department'] = course_master_data['Department Name'].str[:4]
course_master_data['Course_Level'] = course_master_data['Code'].str[2:3]
course_master_data['Program'] = course_master_data['Department'] + course_master_data['Course_Level']
course_master_data = course_master_data[course_master_data['Duration'] != '0 days 00:15:00' & ~course_master_data['Course_Level'].isin(['D', 'E'])]
course_master_data = course_master_data[~course_master_data['Course_Level'].isin(['D', 'E'])]

def sort_key(x):
    try:
        return (0, float(x), '')
    except ValueError:
        return (1, 0, x)

def sort_and_combine(series):
    all_values = []
    for val in series:
        if not isinstance(val, str) or val.strip() == 'nan' or val.strip() == '':
            continue
        all_values.extend([v.strip() for v in val.split(',')])

    unique_vals = sorted(set(all_values), key=sort_key)
    return ', '.join(unique_vals) if all_values else ''

course_master_data['AT Weeks'] = course_master_data.groupby(['Code', 'Type', 'Group'], as_index=False)['AT Weeks'].transform(sort_and_combine)
course_master_data['WT Weeks'] = course_master_data.groupby(['Code', 'Type', 'Group'], as_index=False)['WT Weeks'].transform(sort_and_combine)

course_df = course_master_data[['Code', 'Type', 'Group', 'Department Name', 'Module', 'Description', 'Duration', 'AT Weeks', 'Staff', 'NUMSTU', 'Department', 'Course_Level', 'Program', 'Sample']].copy()
course_df = course_df[course_df['Sample'] == True]
course_df['Key'] = course_df['Code'] + '_' + course_df['Group']
course_df = course_df.drop_duplicates()

staff_df = course_master_data[['Staff', 'Code']].copy()

staff_df['First Name'] = staff_df['Staff'].str.split(',').str[0].str.strip()
staff_df['Title'] = staff_df['Staff'].str.split(',').str[1].str.strip().str.split().str[0]
staff_df['Last Name'] = staff_df['Staff'].str.split(',').str[1].str.strip().str.split().str[1:].str.join(' ')

staff_df = staff_df.dropna().drop_duplicates()