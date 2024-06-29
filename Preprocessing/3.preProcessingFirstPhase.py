from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

files_attack = ["temp/attacksOnIED1AfromSCADA.csv",
                "temp/attacksOnIED1BfromSCADA.csv",
                "temp/attacksOnIED4CfromSCADA.csv",
                "temp/attacksOnHMIfromIED1B.csv"]

files_benign = ["temp/benignIED1A/modbus.csv",
                "temp/benignIED1B/modbus.csv",
                "temp/benignIED4C/modbus.csv",
                "temp/benignHMI/modbus.csv"]

file = "temp/merged/modbusDataset.csv"
oe_save = "processed/encoders/oe_modbusDataset.gz"
df_save = "processed/firstPhase/modbusDataset.csv"
co_save = "processed/firstPhase/corr/modbusDataset.png"

separator = ','

df_benign0 = pd.read_csv(files_benign[0], sep=separator, header=0, index_col=None)
df_benign1 = pd.read_csv(files_benign[1], sep=separator, header=0, index_col=None)
df_benign2 = pd.read_csv(files_benign[2], sep=separator, header=0, index_col=None)
df_benign3 = pd.read_csv(files_benign[3], sep=separator, header=0, index_col=None)
df_benign = pd.concat([df_benign0, df_benign1, df_benign2, df_benign3], ignore_index=True)

del df_benign0
del df_benign1
del df_benign2
del df_benign3

df_attack0 = pd.read_csv(files_attack[0], sep=separator, header=0, index_col=None)
df_attack1 = pd.read_csv(files_attack[1], sep=separator, header=0, index_col=None)
df_attack2 = pd.read_csv(files_attack[2], sep=separator, header=0, index_col=None)
df_attack3 = pd.read_csv(files_attack[3], sep=separator, header=0, index_col=None)
df_attack = pd.concat([df_attack0, df_attack1, df_attack2, df_attack3], ignore_index=True)

del df_attack0
del df_attack1
del df_attack2
del df_attack3

df = pd.concat([df_attack, df_benign], ignore_index=True)
df.to_csv("temp/merged/modbusDataset.csv", index=False)

del df_attack
del df_benign
print("Merging done.")

# colonne senza celle vuote e con un solo valore, oppure tutte vuote
columns_to_drop = ['proto_name', 'ip_version', 'ip_ihl', 'ip_tos', 'ip_flags', 'ip_frag', 'ip_ttl', 'tcp_dataofs',
                   'tcp_reserved', 'tcp_flags', 'tcp_urgptr', 'tcp_option_mss', 'tcp_option_sackok',
                   'tcp_option_nop_count', 'tcp_option_wscale', 'tcp_option_sack_left', 'tcp_option_sack_right',
                   'modbus_proto_id', 'modbus_unit_id']

# colonne che sono solo indicative di informazioni non correlate al data_type
# per esempio il tempo o sorgente e destinazione
columns_to_drop += ['timestamp', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 'ip_id', 'tcp_seq', 'tcp_ack',
                    'tcp_option_timestamp', 'tcp_option_timestamp_echoreply', 'modbus_trans_id']

df = df.drop(columns=columns_to_drop, axis=1)
print("Columns dropped.")

# attack_type conta solo quando data_type = 1, quindi il resto lo posso riempire come voglio
df['attack_type'] = df['attack_type'].fillna("Benign")

indicator = MissingIndicator(error_on_new=True, features='missing-only')
temp = indicator.fit_transform(df)
indicator_columns = [column + '_NA_IND' for column in df.columns[indicator.features_]]
indicator_df = (pd.DataFrame(temp, columns=indicator_columns)).astype(int)
df = pd.concat([df, indicator_df], axis=1)

# colonne con più valori, ma che appaiono per un solo tipo di data_type, quindi non importa il valore in sè
# ma solo il fatto che ci sia o meno
columns_to_swap = ['modbus_except_code', 'modbus_register_value', 'modbus_register_addr']
# colonne il cui missing indicator ha una correlazione più alta col data_type
columns_to_swap += ['modbus_start_addr', 'modbus_output_addr', 'modbus_output_value', 'modbus_byte_count',
                    'modbus_register_val']
# colonne in cui si perde un po' di correlazione ma non troppa
# e che non sono correlate ad altre feature del dataset
# non ci sono altre operazioni che si possono fare senza perdere completamente quell'informazione
# quindi si fa così perchè almeno qualcosa rimane
columns_to_swap += ['modbus_quantity']
# queste colonne da sole hanno una correlazione molto alta con data_type
# il problema è che ci sono valori nulli e qualsiasi operazione che si faccia abbassa tantissimo la correlazione
# queste feature hanno una correlazione alta con altre feature del dataset
columns_to_swap += ['modbus_coil_status_0', 'modbus_coil_status_1', 'modbus_input_status_0',
                    'modbus_input_status_1']

for col in columns_to_swap:
    cols = list(df.columns)
    a, b = cols.index(col), cols.index(col + '_NA_IND')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    df = df.drop(columns=col, axis=1)
print("Columns swapped.")

# prima si faceva anche src_ip, dst_ip, src_mac, dst_mac
columns_to_encode = ['flow', 'modbus_type', 'attack_type']
enc = OrdinalEncoder()
enc.fit(df[columns_to_encode])
joblib.dump(enc, oe_save)
# enc = joblib.load('processed/oe_modbusDataset.gz') # per caricare
df[columns_to_encode] = enc.transform(df[columns_to_encode])
print("Encoding done.")

correlation_matrix = df.corr()
plt.figure(figsize=(50, 50))
sns.set(font_scale=1)
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.savefig(co_save)
plt.close()
print("Saved correlation matrix.")

df.to_csv(df_save, index=False)
del df
