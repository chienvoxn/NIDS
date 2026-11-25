from pathlib import Path
import hashlib
import io
import json
from time import perf_counter
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from nids.soc_charts import style, matrix_chart, batch_summary, graph_chart
from nids.demo_engine import DemoEngine
from nids.demo_lab import scenario, SCENARIOS, replay, evaluate_predictions, topology, topology_examples
from nids.lab_evidence import mixed_case
ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='NIDS Security Center', page_icon='🛡️', layout='wide')
st.markdown('''<style>
.stApp {background:#0b1120;color:#e2e8f0} [data-testid="stSidebar"] {background:#111c2e}
[data-testid="stMetric"] {background:#162338;border:1px solid #304158;border-radius:12px;padding:16px}
[data-testid="stMetricValue"] {color:#38bdf8} h1,h2,h3 {letter-spacing:-.025em}
</style>''',unsafe_allow_html=True)
st.title('🛡️ NIDS Security Center')
st.caption('CNN + MHSA (primary)  /  Flow CSV  /  CPU  /  Authorized lab')
mode=st.sidebar.radio('Chức năng',['CSV Detection','Attack Scenario Replay','Botnet Topology','Stream Monitoring'])
st.sidebar.caption('Không lưu PCAP • Không dùng GeneratedLabelledFlows')

if mode=='Stream Monitoring':
    from stream_view import render_stream
    render_stream(ROOT)
    st.stop()

def figure(w=8,h=3):
    fig,ax=plt.subplots(figsize=(w,h),facecolor='#162338'); ax.set_facecolor('#162338')
    ax.tick_params(colors='#cbd5e1'); ax.xaxis.label.set_color('#cbd5e1'); ax.yaxis.label.set_color('#cbd5e1')
    for spine in ax.spines.values(): spine.set_color('#475569')
    return fig,ax

def show(fig):
    st.pyplot(fig,use_container_width=True); plt.close(fig)

def network_view(g, edges):
    if len(g)>100 or len(edges)>20000 or g.number_of_edges()>200:
        st.info('Graph too large; see host degree table.'); return
    st.plotly_chart(graph_chart(g,edges),use_container_width=True)

if mode=='Botnet Topology':
    st.header('Botnet intelligence')
    st.caption('Phân tích topology và chu kỳ giao tiếp; không xác nhận host nhiễm botnet.')
    case=st.selectbox('Dữ liệu mô phỏng',list(topology_examples()))
    upload=st.file_uploader('CSV tùy chọn: src,dst,time (ISO timestamp)',type=['csv'])
    try:
        df=None
        if upload is not None:
            if upload.size>10*1024*1024: raise ValueError('Limit 10 MiB')
            df=pd.read_csv(upload,nrows=20001,dtype={'src':str,'dst':str})
            if len(df)>20000 or df.empty or df[['src','dst','time']].isna().any().any(): raise ValueError('Invalid metadata or too many rows')
            df['time']=pd.to_datetime(df.time,utc=True,errors='raise'); edges=list(zip(df.src,df.dst))
            st.caption('Nguồn: CSV tải lên (thay thế topology mẫu).')
        else:
            edges=topology_examples()[case]; st.caption('Nguồn: topology mô phỏng tích hợp.')
        label,g=topology(edges)
        a,b,c=st.columns(3); a.metric('Hosts',len(g)); b.metric('Unique links',g.number_of_edges()); c.metric('Topology',label)
        left,right=st.columns([3,2]);
        with left:
            st.subheader('Communication graph'); network_view(g,edges)
        with right:
            st.subheader('Host degree'); st.dataframe(pd.DataFrame(list(g.degree()),columns=['host','degree']).sort_values('degree',ascending=False),hide_index=True,height=260)
        if df is not None:
            from nids.botnet import beaconing_features
            candidates=beaconing_features(df)
            st.divider(); st.subheader('Beaconing / IAT analysis')
            if candidates.empty: st.info('Không đủ kết nối để tính IAT candidate.')
            else:
                pairs=list(candidates[['src','dst']].itertuples(index=False,name=None)); src,dst=st.selectbox('Cặp giao tiếp',pairs,format_func=lambda x:f'{x[0]} → {x[1]}')
                ts=df.loc[(df.src==src)&(df.dst==dst),'time'].sort_values(); gaps=ts.diff().dt.total_seconds().dropna(); mean=float(gaps.mean()); sd=float(gaps.std(ddof=0)); cv=sd/mean if mean>0 else None
                a,b,c,d=st.columns(4); a.metric('Connections',len(ts)); b.metric('Mean IAT',f'{mean:.2f} s'); c.metric('IAT Std (population)',f'{sd:.2f} s'); d.metric('IAT CV','N/A' if cv is None else f'{cv:.3f}')
                left,right=st.columns(2)
                seconds=(ts-ts.iloc[0]).dt.total_seconds()
                with left:
                    st.subheader('Connection timeline')
                    f=go.Figure(go.Scatter(x=seconds,y=[1]*len(seconds),mode='markers',marker=dict(size=11,color='#38bdf8'),hovertemplate='%{x:.2f} s<extra>Event</extra>'))
                    f.update_xaxes(title='Seconds since first event'); f.update_yaxes(visible=False,range=[0,2]); st.plotly_chart(style(f,260),use_container_width=True)
                with right:
                    st.subheader('Interval stability')
                    f=go.Figure(go.Scatter(x=list(range(1,len(gaps)+1)),y=gaps,mode='lines+markers',line=dict(color='#34d399')))
                    f.update_xaxes(title='Interval index',dtick=1); f.update_yaxes(title='IAT (s)',rangemode='tozero'); st.plotly_chart(style(f,260),use_container_width=True)
                st.caption('Khoảng thời gian đều chỉ là tín hiệu tham khảo; heartbeat hợp lệ cũng có thể định kỳ. Thẻ Std dùng độ lệch chuẩn tổng thể; bảng gốc giữ nguyên thuật toán.')
                st.dataframe(candidates,hide_index=True); st.download_button('Export IAT candidates',candidates.to_csv(index=False),'iat_candidates.csv','text/csv')
        else: st.info('Upload CSV có timestamp để xem Beaconing Timeline.')
    except Exception as exc: st.error(str(exc))
    st.stop()

registry=json.loads((ROOT/'configs/demo_models.json').read_text(encoding='utf-8'))
name=st.sidebar.selectbox('Model',list(registry)); cfg=registry[name]
@st.cache_resource
def load_engine(path,dataset,profile,variant,revision): return DemoEngine(path,dataset,profile,variant=variant)
try:
    folder=ROOT/cfg['directory']; variant=cfg['variant']; required=[folder/'cnn_preprocessing.joblib',folder/('cnn_metrics.json' if cfg['profile']=='full' else 'tuning_metrics.json'),folder/(f'{variant}.pt' if cfg['profile']=='full' else f'{variant}_best_state.pt')]
    if not all(x.is_file() for x in required): st.info('Model artifacts not available yet.'); st.stop()
    revision=tuple((x.stat().st_mtime_ns,x.stat().st_size) for x in required)
    engine=load_engine(str(folder),cfg['dataset'],cfg['profile'],variant,revision)
    st.header(mode)
    if cfg['profile']=='tuning': st.warning('Checkpoint tạm thời, chọn bằng validation.')
    with st.expander('Model metadata / feature order'): st.json(engine.metadata); st.write(engine.features)
    upload=st.file_uploader('CSV đúng schema huấn luyện',type=['csv'])
    if upload is None: st.stop()
    if upload.size>50*1024*1024: raise ValueError('Limit 50 MiB')
    raw=upload.getvalue(); frame=pd.read_csv(io.BytesIO(raw),nrows=20001)
    if len(frame)>20000 or frame.empty: raise ValueError('Empty CSV or limit 20,000 flows')
    case=None
    if mode=='Attack Scenario Replay':
        if cfg['dataset']!='cic': st.info('Scenario presets require CIC.'); st.stop()
        st.info('CSV replay mô phỏng; không bắt packet. Thời gian xử lý không phải detection latency mạng.')
        case=st.selectbox('Kịch bản',['PortScan + BENIGN','DoS Hulk + BENIGN','Bot + BENIGN',*SCENARIOS]); batch=st.slider('Flows per batch',1,128,32)
        frame=mixed_case(frame,case.split(' + ')[0]) if case.endswith(' + BENIGN') else scenario(frame,case)
    else: batch=128
    st.caption(f'{len(frame):,} flows ready • batch {batch}')
    key=(mode,name,revision,hashlib.sha256(raw).hexdigest(),case,batch)
    if st.button('Run detection',type='primary'):
        progress=st.progress(0); pieces=[]; start=perf_counter(); count=0
        for chunk in replay(engine,frame,batch):
            pieces.append(chunk); count+=len(chunk); progress.progress(count/len(frame))
        result=pd.concat(pieces,ignore_index=True); elapsed=perf_counter()-start
        st.session_state['soc_result']=(key,result,elapsed)
    saved=st.session_state.get('soc_result')
    if saved is None or saved[0]!=key: st.stop()
    _,result,elapsed=saved
    if case: st.subheader(f'Scenario: {case}')
    report=evaluate_predictions(result,engine.labels,engine.normal)
    a,b,c,d=st.columns(4); a.metric('Flows',len(result)); b.metric('Attack predictions',int(result.is_attack.sum())); c.metric('Benign predictions',int((~result.is_attack).sum())); d.metric('Wall ms/flow',f'{elapsed*1000/len(result):.2f}')
    overview,metrics,alerts=st.tabs(['Traffic overview','Evaluation','Alerts & export'])
    with overview:
        counts=result.predicted.value_counts().sort_values()
        f=go.Figure(go.Bar(x=counts.values,y=counts.index,orientation='h',text=counts.values,textposition='auto',marker_color=['#34d399' if x==engine.normal else '#38bdf8' for x in counts.index]))
        f.update_xaxes(title='Flows'); st.plotly_chart(style(f,max(260,min(470,25*len(counts)+100))),use_container_width=True)
        if mode=='Attack Scenario Replay':
            bs=batch_summary(result,batch); f=go.Figure()
            for col,color in [('Attack predictions','#fb7185'),('Benign predictions','#34d399')]: f.add_bar(x=bs['Batch index'],y=bs[col],name=col,marker_color=color)
            f.update_layout(barmode='stack'); f.update_xaxes(title='Batch index',dtick=1); st.plotly_chart(style(f,280),use_container_width=True)
            st.caption('Batch cuối có thể ít flow hơn. Counts are predictions, not detection rates.')
            st.dataframe(bs,hide_index=True)
    with metrics:
        if report:
            a,b,c,d=st.columns(4); a.metric('Accuracy',f'{report["accuracy"]:.2%}'); b.metric(f'F1 ({report["average"]})',f'{report["f1"]:.2%}'); c.metric('FPR','N/A' if report['fpr'] is None else f'{report["fpr"]:.2%}'); d.metric('Detection rate','N/A' if report['detection_rate'] is None else f'{report["detection_rate"]:.2%}')
            st.caption('Metrics trên mẫu upload/replay; không thay thế test chính thức.')
            binary=pd.DataFrame([[report['tn'],report['fp']],[report['fn'],report['tp']]],index=['Normal','Attack'],columns=['Normal','Attack'])
            mat=report['matrix']; active=(mat.sum(axis=0)+mat.sum(axis=1))>0
            percent=st.toggle('Normalize by actual class',value=False)
            left,right=st.columns([1,2])
            with left:
                st.subheader('Binary matrix'); st.plotly_chart(matrix_chart(binary,percent),use_container_width=True)
            with right:
                st.subheader('Multiclass matrix'); st.plotly_chart(matrix_chart(mat.loc[active,active],percent),use_container_width=True)
            st.caption('Rows without actual samples have undefined recall; normalized empty rows shown as zero.')
            st.dataframe(report['per_class'],hide_index=True); st.json({k:report[k] for k in ('precision','recall','tn','fp','fn','tp')}); st.download_button('Export metrics',json.dumps({k:v for k,v in report.items() if k not in ('matrix','per_class')},indent=2),'metrics.json','application/json')
        else: st.info('CSV không có nhãn thực tế.')
    with alerts:
        options=sorted(result.predicted.unique()); chosen=st.multiselect('Filter predicted class',options,default=options); only=st.checkbox('Only attack predictions',value=True); filtered=result[result.predicted.isin(chosen)]; filtered=filtered[filtered.is_attack] if only else filtered
        st.dataframe(filtered,hide_index=True); st.download_button('Export all predictions',result.to_csv(index=False),'predictions.csv','text/csv'); st.caption('Confidence = uncalibrated max softmax. processed_at_utc là thời gian xử lý, không phải timestamp lưu lượng.')
except Exception as exc: st.error(str(exc))
