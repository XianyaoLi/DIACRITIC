"""Vector figures for the theory-led revision.

Default: rebuild from the accompanying, auditable numeric snapshot.
--refresh-data: extract that snapshot from the original per-step result ledgers.
All displayed rate means, seed counts and uncertainty bands are computed here.
"""
from pathlib import Path
import argparse
import json
import math
import os
from collections import defaultdict

os.environ.setdefault('MPLCONFIGDIR', '/tmp/diacritic-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'isaac' / 'cluster_results' / 'results'
DATA = HERE / 'restructured_figure_data.json'
BLUE, ORANGE, TEAL, GRAY = '#245b8e', '#b85a24', '#277b68', '#69727a'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})

def records(names):
    out = []
    for name in names:
        seen = {}
        for line in (RAW / (name + '.jsonl')).open():
            r = json.loads(line)
            a = r['args']
            model = Path(a.get('save_model') or r.get('model') or '').name
            ds = Path(r.get('data') or a['data']).name
            key = (model, ds, a['seed'], a.get('variant'), a.get('beta'),
                   a.get('lam'), a.get('K'), a.get('aux_theta'), a.get('sym_input'))
            seen[key] = dict(r, _ledger=name+'.jsonl', _dataset=ds, _model=model)
        out.extend(seen.values())
    return out

def gate(r):
    p, phases = r['per_t'], r['phases']
    if r['_dataset'].startswith('taskA'):
        vals = [p['S_G'][i] for i,ph in enumerate(phases) if ph == 'place']
        return float(np.nanmean(vals)) > .9
    hg = p.get('H(Gam|O)', [0 if math.isnan(x) else 1 for x in p['S_Gam']])
    ha = p.get('H(G|O)', [0 if math.isnan(x) else 1 for x in p['S_G']])
    mem = [i for i,ph in enumerate(phases) if ph in ('gap1','gap2') and hg[i] > 1e-9]
    dec = []
    for ph in ('grasp', 'place'):
        ids = [i for i,q in enumerate(phases) if q == ph and ha[i] > 1e-9]
        if ids:
            dec.append(ids[0])
    return bool(mem) and all(p['S_Gam'][i] > .9 for i in mem) and all(p['S_G'][i] > .9 for i in dec)

def phase_rate(r, phase):
    return float(np.nanmean([v for v,ph in zip(r['per_t']['H(C|O)'],r['phases']) if ph == phase]))

def compact(r):
    p = r['per_t']
    return dict(ledger=r['_ledger'], dataset=r['_dataset'], model=r['_model'],
                seed=r['args']['seed'], beta=r['args']['beta'], sufficient=gate(r),
                phases=r['phases'], HC=p['H(C|O)'], HG=p['H(G|O)'], HGamma=p['H(Gam|O)'])

def refresh():
    task = records(['taskA','sysK','taskA_big'])
    task = [r for r in task if r['args'].get('variant') == 'diacritic'
            and r['args'].get('n_train') == 448 and r['args']['beta'] == .001
            and not r['args'].get('aux_mik')]
    world = []
    for m in (4,8,16,32,128,512):
        rs = [r for r in task if r['_dataset'] == f'taskA2_M{m}']
        plain = [r for r in rs if r['args']['K'] == 16 and not r['args'].get('aux_theta')]
        k = max(r['args']['K'] for r in rs if r['args'].get('aux_theta'))
        sys = [r for r in rs if r['args']['K'] == k and r['args'].get('aux_theta')]
        assert len(plain) == len(sys) == 8, (m,len(plain),len(sys))
        world.append(dict(task='Task A',M=m,required=0,sys_K=k,
            plain=[phase_rate(r,'grasp') for r in plain if gate(r)],
            sys=[phase_rate(r,'grasp') for r in sys],plain_total=len(plain),
            sources=[r['_ledger']+':'+r['_model'] for r in plain+sys]))
    for m in (32,128,512):
        rs = records([f'tf_readoutb2_M{m}'])
        k = max(r['args']['K'] for r in rs if r['args'].get('aux_theta'))
        plain = [r for r in rs if r['args']['K'] == 16 and not r['args'].get('aux_theta') and '_distill' not in r['_model']]
        forecast = [r for r in rs if r['args']['K'] == 16 and not r['args'].get('aux_theta') and '_distillforecast' in r['_model']]
        sys = [r for r in rs if r['args']['K'] == k and r['args'].get('aux_theta')]
        assert len(plain) == len(forecast) == len(sys) == 8, (m,len(plain),len(forecast),len(sys))
        world.append(dict(task='Readout-2',M=m,required=2,sys_K=k,
            plain=[phase_rate(r,'gap1') for r in plain if gate(r)],
            forecast=[phase_rate(r,'gap1') for r in forecast if gate(r)],
            sys=[phase_rate(r,'gap1') for r in sys],plain_total=len(plain),
            sources=[r['_ledger']+':'+r['_model'] for r in plain+forecast+sys]))
    prof = [r for r in records(['e1e4']) if r['_dataset']=='tier4_gap20'
            and r['args']['beta']==.001 and r['args']['sym_input']==2]
    assert len(prof) == 8 and all(gate(r) for r in prof)
    plain = [r for r in records(['fig3']) if r['args'].get('variant')=='diacritic'
             and r['args']['beta'] in (0,.001) and r['args'].get('lam',0)==0]
    assert len(plain)==16, len(plain)
    data=dict(provenance='Original per-step ledgers; last record per model/seed; full gap-and-first-decision gate.',
              external=external_cells(),world=world,matched=[compact(r) for r in prof],plain=[compact(r) for r in plain])
    DATA.write_text(json.dumps(data,indent=2,allow_nan=True)+'\n')
    return data

def save(fig, name):
    fig.savefig(HERE/(name+'.pdf'),bbox_inches=None)
    fig.savefig(HERE/(name+'.svg'),bbox_inches=None)
    fig.savefig(HERE/(name+'.png'),dpi=220,bbox_inches=None)
    plt.close(fig)

def concept():
    fig=plt.figure(figsize=(5.5,2.5))
    ax=fig.add_axes([.01,.015,.98,.97]);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    def box(x,y,w,h,color,title,subtitle):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.4,rounding_size=1.2',
                                   fc=color,ec='#8c969e',lw=.65))
        ax.text(x+w/2,y+h*.67,title,ha='center',va='center',fontsize=8.3,color='#162b3c')
        ax.text(x+w/2,y+h*.28,subtitle,ha='center',va='center',fontsize=6.8,color='#3d4952')
    box(.8,62,23,23,'#f1f3f5',r'History $H_t$','observations + actions')
    box(34,78,42,20,'#e7eff7',r'Current class $G_{E,t}$','distinctions required by the current action')
    box(34,48,42,24,'#d5e6f6',r'Behavioral memory $\Gamma_t$','distinctions that must persist\ngiven future observations')
    box(82,48,17.2,24,'#f8e7d8',r'Code $C_t$','learned\nsole carrier')
    def arr(a,b,**kwargs):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.9,color='#51616e',**kwargs))
    arr((24,78),(33,88))
    arr((24,67),(33,60))
    ax.text(27.4,91,'now',ha='center',fontsize=6.5,color=GRAY)
    ax.text(27.4,50,'across\ntime',ha='center',fontsize=6.1,color=GRAY)
    arr((76.5,60),(81.5,60));ax.text(79,64,'learn',ha='center',fontsize=6.5,color=GRAY)
    ax.text(50,39,'Re-observed before use: no need to carry it across the wait (re-reveal toy, §4.3).',
            ha='center',va='center',fontsize=7.1,color=TEAL)
    ax.plot([0,100],[32,32],color='#c1c8cd',lw=.65)
    xs=[39,52.5,66,79.5,93]
    labs=['wait','first grasp','wait','first place','done']
    ax.text(0,26,"A' after the cue",fontweight='bold',fontsize=7.4,va='center')
    ax.text(0,15,r'$H(\Gamma_t\mid O_t)$ [bits]',fontsize=8,color=BLUE,va='center')
    ax.text(0,4,r'$H(G_{E,t}\mid O_t)$ [bits]',fontsize=8,color=ORANGE,va='center')
    for x,lab,g,now in zip(xs,labs,[2,2,1,1,0],[0,1,0,1,0]):
        if lab.startswith('first'):
            ax.add_patch(Rectangle((x-6.2,-1),12.4,31,fc='#fbf3ec',ec='none',zorder=-1))
        ax.text(x,26,lab,ha='center',va='center',fontsize=7)
        ax.text(x,15,str(g),ha='center',va='center',fontsize=9,fontweight='bold',color=BLUE)
        ax.text(x,4,str(now),ha='center',va='center',fontsize=9,color=ORANGE)
    save(fig,'fig1_behavioral_memory')

def external_cells():
    """bsuite memory chain (environment unmodified, MIKASA-Base): certified requirement and learned rate of sufficient seeds per (bits, delay)."""
    ext = HERE.parent / 'external' / 'results'
    runs = {}
    for f in [ext / 'cert.jsonl'] + sorted((ext / 'cluster').glob('cert.*.jsonl')):
        for line in f.open():
            r = json.loads(line)
            if r.get('kind') == 'run' and r['task'] == 'memchain': runs[(r['bits'], r['L'], r['mode'], r['seed'])] = r
    cells = []
    for b in (1, 2, 3):
        for L in (10, 30, 100):
            g = [r for k, r in runs.items() if k[0] == b and k[1] == L]
            assert len(g) == 16, (b, L, len(g))
            cells.append(dict(bits=b, delay=L, required=float(np.mean([r['req_mid'] for r in g])),
                              plain=[r['rate_mid'] for r in g if r['mode'] == 'plain' and r['sufficient']],
                              supervised=[r['rate_mid'] for r in g if r['mode'] == 'generic' and r['sufficient']], runs=len(g)))
    return cells

def world(data):
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.45),gridspec_kw=dict(width_ratios=[1.18,.92,1.0]))
    fig.subplots_adjust(left=.075,right=.995,bottom=.34,top=.88,wspace=.3)
    for ax,task,title in zip(axes[:2],['Task A','Readout-2'],['(a) Task A: 0 bits','(b) Readout-2: 2 bits']):
        rows=[r for r in data['world'] if r['task']==task];x=np.log2([r['M'] for r in rows])
        for key,col,marker,label in [('sys',ORANGE,'^','system identification'),('plain',BLUE,'o','plain behavioral learner'),('forecast',TEAL,'s','future-behavior supervision')]:
            if key not in rows[0]:continue
            ax.errorbar(x,[np.mean(r[key]) for r in rows],[np.std(r[key]) for r in rows],
                        color=col,marker=marker,ms=3.5,lw=1,elinewidth=.65,capsize=1.7,label=label)
        ax.axhline(rows[0]['required'],color='#151515',lw=.9,ls=(0,(4,3)),label='certified requirement',zorder=0)
        if task=='Task A':      # all six cells are 8/8: one line instead of six colliding labels
            assert all(len(r['plain'])==r['plain_total'] for r in rows)
            ax.text(.03,.13,f"{rows[0]['plain_total']}/{rows[0]['plain_total']} sufficient at every $M$",transform=ax.transAxes,color=BLUE,fontsize=6.6,ha='left')
        else:
            for i,(xx,r) in enumerate(zip(x,rows)):    # end labels are anchored inwards so they clear the y-axis and the frame
                ha,dx=('left',-.12) if i==0 else ('right',.12) if i==len(rows)-1 else ('center',0)
                ax.text(xx+dx,rows[0]['required']+.75,f"{len(r['plain'])}/{r['plain_total']}",color=BLUE,fontsize=6.6,ha=ha)
            ax.text(.98,.95,'supervised: 8/8 each',ha='right',va='top',transform=ax.transAxes,color=TEAL,fontsize=6.6)
        ax.set_title(title,loc='left',pad=6)
        ax.set_xticks(x);ax.set_xticklabels([str(r['M']) for r in rows]);ax.set_xlabel('hidden modes $M$',labelpad=2)
        ax.set_ylim(-.25,9.5);ax.set_yticks([0,2,4,6,8]);ax.grid(axis='y',alpha=.15,lw=.5)
    axes[0].set_ylabel('code rate (bits)',labelpad=3)
    # (c) external benchmark: every sufficient seed matches the requirement at mid-delay
    ax=axes[2];cells=data['external'];delays=sorted({c['delay'] for c in cells});pos={d:i for i,d in enumerate(delays)}
    for b in (1,2,3):
        cs=sorted([c for c in cells if c['bits']==b],key=lambda c:c['delay'])
        ax.plot([pos[c['delay']] for c in cs],[c['required'] for c in cs],color='#151515',lw=.9,ls=(0,(4,3)),zorder=1)
        ax.text(2.40,cs[-1]['required'],f"{b} bit"+("s" if b>1 else ""),fontsize=6.4,va='center',ha='left',color='#151515')
        for c in cs:
            for key,col,marker,off in (('plain',BLUE,'o',-.10),('supervised',TEAL,'s',.10)):
                v=c[key]
                if v:ax.plot(pos[c['delay']]+off+np.linspace(-.06,.06,len(v)) if len(v)>1 else [pos[c['delay']]+off],v,ls='none',marker=marker,ms=3.1,color=col,mec='white',mew=.3,zorder=3)
                # sufficient-seed count per learner (8 runs each); the overlapping markers cannot be counted by eye
                ax.text(pos[c['delay']]+1.8*off,c['required']+.13,f"{len(v)}/{c['runs']//2}",color=col,fontsize=5.6,ha='center',va='bottom')
            if not c['plain'] and not c['supervised']:
                ax.plot([pos[c['delay']]],[c['required']],ls='none',marker='o',ms=4.2,mfc='white',mec=GRAY,mew=.8,zorder=3)
    ax.set_title('(c) Memory chain',loc='left',pad=6)
    ax.text(.03,.04,'mid-delay rate\nexternal, unmodified',transform=ax.transAxes,color=GRAY,fontsize=6.6,ha='left')
    ax.set_xticks(range(len(delays)));ax.set_xticklabels([str(d) for d in delays]);ax.set_xlabel('delay (steps)',labelpad=2)
    ax.set_xlim(-.45,3.0);ax.set_ylim(-.1,3.6);ax.set_yticks([0,1,2,3]);ax.grid(axis='y',alpha=.15,lw=.5)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.005),ncol=2,frameon=False,columnspacing=1.3,handlelength=1.7)
    save(fig,'fig2_world_separation')

def trajectory(ax,rows,title,plain=False):
    r=rows[0];x=np.arange(1,len(r['phases'])+1)
    for q in rows:
        ax.plot(x,q['HC'],color=BLUE if q['sufficient'] else '#bd5a61',lw=.75,alpha=.6,
                label='learned code, sufficient' if q['sufficient'] else 'learned code, insufficient')
    ax.step(x,r['HGamma'],where='mid',color='#111111',ls=(0,(4,2)),lw=1.25,label=r'$H(\Gamma_t\mid\bar O_t)$')
    ax.step(x,r['HG'],where='mid',color=ORANGE,ls=(0,(1.4,1.4)),lw=1.1,label=r'$H(G_{E,t}\mid\bar O_t)$')
    ymax=4.35 if plain else 2.65
    for phase,lab in [('scan','cue'),('gap1','wait'),('grasp','grasp'),('gap2','wait'),('place','place')]:
        ids=[i+1 for i,p in enumerate(r['phases']) if p==phase]
        if not ids:continue
        if phase in ('gap1','gap2'):ax.axvspan(min(ids)-.5,max(ids)+.5,color='#e9eef3',alpha=.65,zorder=-1)
        xc,ha=(min(ids)+max(ids))/2,'center'
        if phase=='scan':xc,ha=max(ids)-.5,'right'    # the cue phase is short: right-align its label so it clears the wait band
        ax.text(xc,ymax-.1,lab,ha=ha,va='top',fontsize=6.5,color=GRAY)
    ax.set(xlim=(.5,len(x)+.5),ylim=(-.08,ymax),xlabel='episode step',ylabel='code rate (bits)')
    ax.set_yticks(range(5) if plain else [0,1,2]);ax.set_title(title,loc='left',pad=5)
    ax.grid(axis='y',alpha=.13,lw=.5)

def profiles(data):
    fig,ax=plt.subplots(figsize=(5.5,2.35));fig.subplots_adjust(left=.075,right=.99,top=.9,bottom=.30)
    trajectory(ax,data['matched'],"Matched side information, gap 20: 8/8 sufficient")
    h,l=ax.get_legend_handles_labels();unique=dict(zip(l,h))
    fig.legend(unique.values(),unique.keys(),loc='lower center',bbox_to_anchor=(.54,.005),frameon=False,ncol=3,handlelength=1.8)
    save(fig,'fig3_memory_profile')
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.35));fig.subplots_adjust(left=.075,right=.99,top=.88,bottom=.35,wspace=.22)
    for ax,b in zip(axs,[0,.001]):
        rows=sorted([r for r in data['plain'] if r['beta']==b],key=lambda x:x['seed'])
        trajectory(ax,rows,fr'$\beta={b:g}$: {sum(r["sufficient"] for r in rows)}/8 sufficient',plain=True)
    axs[1].set_ylabel('')
    h,l=axs[0].get_legend_handles_labels();unique=dict(zip(l,h))
    fig.legend(unique.values(),unique.keys(),loc='lower center',bbox_to_anchor=(.52,.005),frameon=False,ncol=2)
    save(fig,'fig3_plain_seeds')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-data',action='store_true');args=parser.parse_args()
    data=refresh() if args.refresh_data else json.loads(DATA.read_text())
    concept();world(data);profiles(data)
    print('Wrote four PDF/SVG/PNG figures from',DATA)
    for r in data['world']:
        print(r['task'],r['M'],'plain',len(r['plain']),round(float(np.mean(r['plain'])),3),
              'sys',round(float(np.mean(r['sys'])),3))
    for c in data['external']:
        print('memchain',c['bits'],'bit(s) delay',c['delay'],'required',round(c['required'],3),'sufficient plain/supervised',len(c['plain']),len(c['supervised']),
              'max |rate-required|',round(max([abs(v-c['required']) for v in c['plain']+c['supervised']] or [0]),4))
