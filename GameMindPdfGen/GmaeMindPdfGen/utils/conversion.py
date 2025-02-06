import pandas as pd

def normalize_and_correct(df):
    records = []
    for _, row in df.iterrows():
        user = row['user']
        age = row['age']
        session = row['session']

        for col in df.columns:
            if col.startswith('level_') and pd.notna(row[col]):
                level_data = row[col]
                level = col.split('_')[1]
                record = {
                    'User': user,
                    'Session': session,
                    'Level': f'level_{level}',
                    'RandamNumRange': float(level_data.get('randamNumRange', 0)),
                    'Streak': float(level_data.get('streak', 0)),
                    'Time': int(level_data.get('time', 0)),
                    'Age': age
                }
                records.append(record)

    normalized_df = pd.DataFrame.from_records(records)
    normalized_df = normalized_df[['User', 'Session', 'Level', 'RandamNumRange', 'Streak', 'Time', 'Age']]

    return normalized_df
