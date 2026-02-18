import pandas as pd
df = pd.read_csv("data/raw/crisprofft/CRISPRoffT_all_targets.txt", sep='\t', on_bad_lines='skip')
print(df[['Score', 'Indel_treatment%']].head(20))
print("\nScore Non-Null:", df['Score'].notna().sum())
print("Indel% Non-Null:", df['Indel_treatment%'].notna().sum())
print("\nScore Stats:")
print(pd.to_numeric(df['Score'], errors='coerce').describe())
