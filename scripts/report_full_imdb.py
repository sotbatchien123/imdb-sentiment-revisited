"""Generate a clearly marked partial/final report from actual completed runs."""
import json
import statistics
from pathlib import Path
from runtime_io import atomic_json

ROOT=Path(__file__).resolve().parents[1]

def generate_report(cfg):
    run_root=ROOT/cfg['output']; results=[]
    for seed in cfg['seeds']:
        for model in cfg['models']:
            path=run_root/f'{model}_seed{seed}'/'test_metrics.json'
            if path.exists(): results.append(json.loads(path.read_text()))
    expected=len(cfg['seeds'])*len(cfg['models'])
    data=json.loads((ROOT/cfg['dataset']/'manifest.json').read_text(encoding='utf-8'))
    payload={'state':'complete' if len(results)==expected else 'in_progress','completed_runs':len(results),
             'expected_runs':expected,'results':results,'dataset':data['splits'],'aggregate':{}}
    lines=['# Báo cáo thực nghiệm IMDb toàn bộ', '',
           f"Trạng thái: **{'Hoàn thành' if len(results)==expected else 'Đang chạy — chưa phải kết luận cuối'}** ({len(results)}/{expected} lượt).",'',
           '## Phạm vi và dữ liệu','',
           'Dùng 50.000 review có nhãn của IMDb làm nguồn, mọi thể loại; không dùng 50.000 review chưa có nhãn. Test giữ đủ 25.000 review gốc. Các mẫu train trùng/rò rỉ được loại có nhật ký.', '',
           '| Tập | Review | Phim | Tiêu cực | Tích cực |','|---|---:|---:|---:|---:|']
    for split,s in data['splits'].items():
        lines.append(f"| {split} | {s['reviews']} | {s['movies']} | {s['labels'].get('0',0)} | {s['labels'].get('1',0)} |")
    lines += ['',f"Loại khỏi train dẫn xuất: {json.dumps(data['exclusion_counts'],ensure_ascii=False)}.",
              'Không trùng phim hoặc nội dung chuẩn hóa giữa các tập. Test có thể chứa nội dung trùng bên trong; được giữ nguyên theo nguồn.',
              'Một phần test (Horror) đã được đánh giá trước đây; không mô tả đây là holdout hoàn toàn chưa quan sát.', '',
              '## Thiết kế','',
              '- mean: BERT đóng băng + masked mean pooling + Linear; lấy đầu review.',
              '- bilstm: BERT đóng băng + BiLSTM; lấy đầu review.',
              '- bilstm_headtail: cùng BERT và BiLSTM nhưng giữ đầu–cuối review.',
              '- Cùng 256 token, batch 8 × tích lũy 4, AdamW lr 1e-4, 768 hidden/chiều, dropout 0.5.',
              (f"- Tối đa {cfg['max_epochs']} epoch, ít nhất {cfg.get('min_epochs',0)} epoch trước early stopping, patience {cfg['patience']}; chọn best.pt bằng validation Macro-F1." if cfg.get('early_stopping',True) else f"- Chạy đủ {cfg['max_epochs']} epoch, tắt early stopping theo yêu cầu người dùng; chọn best.pt bằng validation Macro-F1, không mặc định dùng epoch cuối."),
              '- Ba seed 42/43/44. Cấu hình đã cố định trước khi xem kết quả test của đợt này.',
              '- So bilstm với bilstm_headtail để đo tác động của cách chọn nội dung; không quy mọi khác biệt với mean pooling cho head–tail.', '',
              '## Kết quả từng lượt','',
              '| Mô hình | Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Thời gian train (phút) |',
              '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['model']} | {r['seed']} | {r['best_epoch']} | {r['accuracy']*100:.2f} | {r['macro_f1']*100:.2f} | {r['training_seconds']/60:.2f} |")
    lines+=['','## Trung bình theo mô hình','','| Mô hình | Số seed | Accuracy (%) | Macro-F1 (%) |','|---|---:|---:|---:|']
    for model in cfg['models']:
        rs=[r for r in results if r['model']==model]
        if not rs: continue
        agg={}
        for metric in ['accuracy','macro_f1']:
            vals=[r[metric] for r in rs]
            agg[metric]={'mean':statistics.mean(vals),'sample_std':statistics.stdev(vals) if len(vals)>1 else None}
        payload['aggregate'][model]=agg
        def cell(metric):
            s=agg[metric]; return f"{s['mean']*100:.2f}"+(f" ± {s['sample_std']*100:.2f}" if s['sample_std'] is not None else ' (mới 1 seed)')
        lines.append(f"| {model} | {len(rs)} | {cell('accuracy')} | {cell('macro_f1')} |")
    if len(results)==expected:
        delta=payload['aggregate']['bilstm_headtail']['macro_f1']['mean']-payload['aggregate']['bilstm']['macro_f1']['mean']
        payload['headtail_minus_head_macro_f1']=delta
        lines+=['',f'Head–tail chênh lệch **{delta*100:+.2f} điểm phần trăm Macro-F1** so với BiLSTM lấy đầu trên tập test này.',
                'Đây là kết quả thực nghiệm; không khẳng định cải thiện nếu chênh lệch âm hoặc bằng 0. Ba seed chưa đủ để khẳng định tính tổng quát trên mọi dữ liệu.']
    else:
        lines+=['','Chưa kết luận phương án cải tiến tốt hơn cho đến khi đủ các lượt đã định.']
    lines+=['','## Review dài và ngắn','','| Mô hình | Seed | Nhóm | Số review | Macro-F1 (%) |','|---|---:|---|---:|---:|']
    for r in results:
        for group,m in r.get('length_groups',{}).items():
            lines.append(f"| {r['model']} | {r['seed']} | {group} | {m['examples']} | {m['macro_f1']*100:.2f} |")
    lines+=['','Nhóm long là review vượt 256 token trước khi chọn nội dung; short không vượt giới hạn.',
            'Thời gian phụ thuộc tải GPU khác nếu có, không được coi là benchmark tốc độ phần cứng có kiểm soát.',
            'Chi tiết classification_report, confusion_matrix và dự đoán từng review nằm trong thư mục mỗi lượt.']
    # Reports are derived from durable per-run results. A viewer holding one open
    # must not stop the remaining training jobs; the next run/report refresh retries.
    try:
        atomic_json(ROOT/'reports/full_imdb_progress_report.json',payload)
    except OSError as exc:
        print(f'JSON report refresh deferred: {exc}',flush=True)
    try:
        (ROOT/'reports/full_imdb_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except OSError as exc:
        print(f'Markdown report refresh deferred: {exc}',flush=True)

if __name__=='__main__':
    generate_report(json.loads((ROOT/'configs/full_imdb.json').read_text()))
