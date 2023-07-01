import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_rows', None)

def transform_data(df):
    for i in range(0, len(df)):
        if isinstance(df['val_acc'][i], list) and len(df['val_acc'][i]) == 3:
            df.loc[i, 'acc'] = round(df.loc[i, 'val_acc'][0], 4)
            df.loc[i, 'val_acc'] = round(df.loc[i, 'val_acc'][2], 4)
        #print(type(df.loc[i, 'adjusted_mIoU']))
        df["adjusted_mIoU"] = df["adjusted_mIoU"].replace('', np.nan).astype(float)

            
        # else:
        #     df.loc[i, 'val_acc'] = None
        # if isinstance(df['val_loss'][i], float) == False:
        #     df['val_loss'][i] = None
    return df

def tmp_trans(df):
    for i in range(0, len(df)):
        if isinstance(df['other_mIoU'][i], list):
            df.loc[i, 'AMIoU_2'] = round(df.loc[i, 'other_mIoU'][0], 4)
        else:
            df.loc[i, 'val_acc'] = 0
            df.loc[i, 'val_loss'] = 0
        # else:
        #     df.loc[i, 'val_acc'] = None
        # if isinstance(df['val_loss'][i], float) == False:
        #     df['val_loss'][i] = None
    df = df.drop('other_mIoU', axis=1)
    return df

Baseline_100epoch = pd.read_pickle('./training_data/Baseline_0.5716_100epoch.pkl')
Baseline_100epoch = Baseline_100epoch.set_index('epoch')
Baseline_100epoch = transform_data(Baseline_100epoch)
Baseline_100epoch = tmp_trans(Baseline_100epoch)

print(Baseline_100epoch)

#Baseline_100epoch_1 = pd.read_pickle('./training_data/Baseline_0.5821_100epoch.pkl')
#print(Baseline_100epoch_1)

only1_vf = pd.read_pickle('./training_data/TOP_18_0.5814_100epoch.pkl')
only1_vf = only1_vf.set_index('epoch')
only1_vf = transform_data(only1_vf)
only1_vf = tmp_trans(only1_vf)
print(only1_vf)

Baseline_100epoch["train_loss"] = Baseline_100epoch["train_loss"] / 744
only1_vf["train_loss"] = only1_vf["train_loss"] / 744
Baseline_100epoch["val_loss"] = Baseline_100epoch["val_loss"] / 125
only1_vf["val_loss"] = only1_vf["val_loss"] / 125
Baseline_100epoch["val_acc"] = Baseline_100epoch["val_acc"] * 100
only1_vf["val_acc"] = only1_vf["val_acc"] * 100


#Baseline_100epoch['train_loss'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
#only1_vf['train_loss'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
Baseline_100epoch['val_acc'].iloc[lambda x: x.index % 2 - 1 == 0].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
only1_vf['val_acc'].iloc[lambda x: x.index % 2 - 1 == 0].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# #train_df_14['val_acc'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# #train_df_15['val_acc'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# train_df_16['val_acc'].plot(linestyle='-', marker='.', linewidth=2)
# train_df_17['val_acc'].plot(linestyle='-', marker='.', linewidth=2)

# # plt.style.use('ggplot')
plt.xlabel('Epoch',  fontsize=18)  # 添加横轴标签
#plt.ylabel('Validation Loss',  fontsize=18)  # 添加纵轴标签
plt.ylabel('MIoU (%)',  fontsize=18)  # 添加纵轴标签
#plt.ylabel('Adjusted MIoU (%)',  fontsize=18)  # 添加纵轴标签
#plt.title('Arcuate Scotoma', fontsize=18)  # 添加标题

# #plt.legend(['Baseline', 'Query', 'Key', 'Value'], loc='lower right')  # 设置图例位置
# plt.legend(['Baseline', '0 & 1', '1 & 2', '1 & 5'], loc='lower right', prop={'size': 18})
plt.legend(['Baseline', 'TANet'], loc='lower right', prop={'size': 18})

# plt.grid(True)  # 显示网格线

plt.tight_layout()  # 调整布局，防止标签重叠
# plt.legend(['Baseline', 'No VF', 'Only1', 'Only3', '1and2', '1and3', 'All VF'])
#plt.savefig('./Thesis_plot/S44_MIoU.png', dpi=300)
plt.show()