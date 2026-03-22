import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy import stats

def iqr_out(s):
    Q1,Q3 = s.quantile([.25,.75])
    return (s<Q1-1.5*(Q3-Q1))|(s>Q3+1.5*(Q3-Q1))

#Bai_1: HOUSING
df = pd.read_csv("ITA105_Lab_2_Housing.csv")
print(df.shape, "\n", df.isnull().sum(), "\n", df.describe())
cols = ['dien_tich','gia','so_phong']
df[cols].plot(kind='box',subplots=True,layout=(1,3),figsize=(12,4),title="B1 Boxplot"); plt.show()
plt.scatter(df.dien_tich,df.gia,alpha=.5); plt.title("B1 Scatter"); plt.show()
for c in cols:
    print(c, "IQR:",iqr_out(df[c]).sum(), "Z:", (np.abs(stats.zscore(df[c]))>3).sum())
df_c=df.copy()
for c in cols:
    Q1,Q3=df[c].quantile(.25),df[c].quantile(.75); IQR=Q3-Q1
    df_c[c]=df_c[c].clip(Q1-1.5*IQR,Q3+1.5*IQR)
df_c[cols].plot(kind='box',subplots=True,layout=(1,3),figsize=(12,4),title="B1 Sau Clip"); plt.show()

#Bai_2: IoT
df2=pd.read_csv("ITA105_Lab_2_Iot.csv",parse_dates=['timestamp']).set_index('timestamp')
print(df2.isnull().sum())
sensors=df2.sensor_id.unique()
fig,ax=plt.subplots(len(sensors),1,figsize=(12,3*len(sensors)))
for i,s in enumerate(sensors):
    sub=df2[df2.sensor_id==s].temperature; ax[i].plot(sub,label=s); ax[i].legend()
plt.suptitle("B2 Temperature"); plt.tight_layout(); plt.show()
for s in sensors:
    sub=df2[df2.sensor_id==s].temperature; rm=sub.rolling(10,min_periods=1).mean(); rs=sub.rolling(10,min_periods=1).std().fillna(0)
    print(f"{s} rolling:", ((sub>rm+3*rs)|(sub<rm-3*rs)).sum())
    for c in ['temperature','pressure','humidity']:
        print(f"  {s} {c} Z:", (np.abs(stats.zscore(df2[df2.sensor_id==s][c]))>3).sum())
df2[['temperature','pressure','humidity']].plot(kind='box',subplots=True,layout=(1,3),figsize=(12,4)); plt.show()
df2_c=df2.copy()
for s in sensors:
    m=df2_c.sensor_id==s
    for c in ['temperature','pressure','humidity']:
        z=np.abs(stats.zscore(df2_c.loc[m,c])); df2_c.loc[m&(z>3),c]=np.nan
    df2_c.loc[m,['temperature','pressure','humidity']]=df2_c.loc[m,['temperature','pressure','humidity']].interpolate()

#Bai_3: E-COMMERCE
df3=pd.read_csv("ITA105_Lab_2_Ecommerce.csv")
print(df3.describe())
ec=['price','quantity','rating']
df3[ec].plot(kind='box',subplots=True,layout=(1,3),figsize=(12,4),title="B3 Boxplot"); plt.show()
for c in ec:
    print(c,"IQR:",iqr_out(df3[c]).sum(),"Z:",(np.abs(stats.zscore(df3[c]))>3).sum())
out=iqr_out(df3.price)|iqr_out(df3.quantity)
plt.scatter(df3[~out].price,df3[~out].quantity,alpha=.5,label='normal')
plt.scatter(df3[out].price,df3[out].quantity,c='red',label='outlier'); plt.legend(); plt.title("B3 Scatter"); plt.show()
print("price=0:",( df3.price==0).sum(),"rating>5:",(df3.rating>5).sum())
df3_c=df3[(df3.price>0)&(df3.rating<=5)].copy()
for c in ['price','quantity']:
    Q1,Q3=df3_c[c].quantile(.25),df3_c[c].quantile(.75); IQR=Q3-Q1
    df3_c[c]=df3_c[c].clip(Q1-1.5*IQR,Q3+1.5*IQR)
df3_c[ec].plot(kind='box',subplots=True,layout=(1,3),figsize=(12,4),title="B3 Sau xử lý"); plt.show()

#Bai_4: MULTIVARIATE
fig,ax=plt.subplots(1,3,figsize=(15,4))
for i,(d,x,y,t) in enumerate([
    (df,'dien_tich','gia','Housing'),
    (df2[df2.sensor_id=='S1'],'temperature','pressure','IoT S1'),
    (df3,'price','quantity','Ecommerce')]):
    u=iqr_out(d[x])|iqr_out(d[y]); m=iqr_out(d[x])&iqr_out(d[y])
    ax[i].scatter(d[~u][x],d[~u][y],s=10,alpha=.5,label='normal')
    ax[i].scatter(d[u&~m][x],d[u&~m][y],c='orange',s=30,label=f'uni({u.sum()})')
    ax[i].scatter(d[m][x],d[m][y],c='red',s=80,marker='*',label=f'multi({m.sum()})')
    ax[i].set_title(t); ax[i].legend(fontsize=7)
plt.suptitle("B4 Multivariate Outlier"); plt.tight_layout(); plt.show()