import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# help function converting text answers into binary (0 or 1) format
# labels are determined manually
def answer_binarize(s):
    bin_dict = {'voted':1, 'too much':0, 'about right': 1, 'too little':0, 
                'too high':0, 'allowed':1, 'favor':1, 'yes':1, 'not allowed':0, 
                'oppose':0, 'should not be legal': 0, 'should be legal': 1, 
                'did not vote':0, 'about right': 1, 'no':0, 'too low':0
                }
    s = s.lower()
    if s in bin_dict.keys():
        return bin_dict[s]
    return None

# import original dataset
with open('data/GSS_by_column.json', 'r') as json_file:
    data = json.load(json_file)

### convert the json format into tabular ###
results_dict = {}
all_rows = []
variables_info = [data['variables'][str(question_index)] for question_index in range(0, 28)]
variable_names = [var['name'] for var in variables_info]

for i in variable_names:
    your_list = data['data'][i]
    distinct_values = set(your_list)
    results_dict[i] = distinct_values

variables_info = [data['variables'][str(question_index)] for question_index in range(0, 28)]
variable_names = [var['name'] for var in variables_info]
questions = [var['Question Text'] for var in variables_info]

num_records = len(data['data']['year'])
for i in range(num_records):
    for var_name, question_text in zip(variable_names, questions):
        row = {
            'year': data['data']['year'][i],
            'yearid': data['data']['id_'][i],
            'variable': var_name,
            'question': question_text,
            'binarized': data['data'][var_name][i] if var_name in data['data'] else None
        }
        all_rows.append(row)


# store the processed data into DataFrame
df = pd.DataFrame(all_rows)
df['binarized'] = df['binarized'].apply(answer_binarize)
df = df.dropna(subset=['binarized'])

# write the cleaned dataframe as parquet file
table = pa.Table.from_pandas(df)
parquet_file = 'data/gss.parquet'
pq.write_table(table, parquet_file)