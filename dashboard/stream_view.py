from pathlib import Path
import pandas as pd
import streamlit as st
from nids.monitor_view import read_snapshot


def render_stream(root):
    st.header('Stream Monitoring')
    st.warning('Replay = mo phong CSV. Shadow = du doan live chua kiem chung. Khong co canh bao live chinh thuc.')
    base=Path(root)/'outputs/stream_detection'
    auto=st.checkbox('Auto refresh (2 seconds)',value=True)
    @st.fragment(run_every=2 if auto else None)
    def panel():
        runs=sorted((p for p in base.iterdir() if p.is_dir()),reverse=True) if base.is_dir() else []
        if not runs:
            st.info('No stream runs. Start scripts/run_stream_detection.py in another terminal.'); return
        choice=st.selectbox('Run', ['Latest']+[p.name for p in runs])
        folder=runs[0] if choice=='Latest' else base/choice
        snap=read_snapshot(folder); summary=snap['summary']; rows=snap['rows']
        st.caption(folder.name)
        status='Completed' if summary and summary.get('complete') else 'Incomplete / writing (not proof of active process)'
        st.caption(status)
        if snap['invalid'] or snap['truncated']: st.warning('Snapshot has invalid records or reached its row limit.')
        if not rows: st.info('Waiting for complete JSONL records.'); return
        df=pd.DataFrame(rows); modes=', '.join(sorted(df['mode'].unique()))
        a,b,c,d=st.columns(4); a.metric('Processed flows',len(df)); b.metric('Attack predictions',int(df.is_attack.sum())); c.metric('Replay alerts',len(snap['alerts'])); d.metric('Mode',modes)
        if 'processed_at_utc' in df:
            counts=df.groupby('processed_at_utc',sort=False).is_attack.agg(['size','sum']).reset_index(); counts.columns=['Processed UTC','Flows','Attack predictions']; counts['Batch index']=range(1,len(counts)+1)
            counts['Benign predictions']=counts['Flows']-counts['Attack predictions']
            st.subheader('Batch activity'); st.bar_chart(counts,x='Batch index',y=['Benign predictions','Attack predictions']); st.caption('Processing timestamps, not network event timestamps.')
        if 'predicted' in df: st.subheader('Prediction distribution'); st.bar_chart(df.predicted.value_counts())
        st.subheader('Replay alerts'); st.dataframe(pd.DataFrame(snap['alerts']).tail(500),hide_index=True)
        st.subheader('Recent predictions (including shadow)'); st.dataframe(df.tail(500),hide_index=True)
        st.download_button('Export snapshot',df.to_csv(index=False),'stream_snapshot.csv','text/csv')
        if summary:
            with st.expander('Run summary / provenance'): st.json(summary)
    panel()
