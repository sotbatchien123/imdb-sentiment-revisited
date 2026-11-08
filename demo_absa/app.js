const $ = id => document.getElementById(id);

const ASPECT_NAMES = {
  acting: 'Diễn xuất', story: 'Cốt truyện', visual: 'Hình ảnh',
  sound: 'Âm thanh', direction: 'Đạo diễn', overall: 'Tổng thể',
};
const ASPECT_HINTS = {
  acting: 'diễn viên, vai diễn, lồng tiếng', story: 'kịch bản, lời thoại, nhân vật, cái kết',
  visual: 'quay phim, ánh sáng, kỹ xảo, bối cảnh', sound: 'nhạc phim, âm thanh',
  direction: 'đạo diễn, cách dàn dựng', overall: 'đánh giá chung về bộ phim',
};
const LABEL_NAMES = {positive: 'Khen', negative: 'Chê', neutral: 'Trung lập', none: 'Không nhắc tới'};
const LABEL_ORDER = ['positive', 'negative', 'neutral', 'none'];

const SAMPLES = {
  'ex-mixed': 'I wanted to love this one. The two leads are wonderful together, and the actress playing the ' +
    'older sister gives a quiet, heartbreaking performance that deserves every award. Unfortunately the script ' +
    'does not trust its audience: every twist is explained twice, the villain has no motivation, and the last ' +
    'thirty minutes drag on forever. The cinematography is lovely, with long golden shots of the coast. ' +
    'Worth seeing for the acting, but I left the theater disappointed.',
  'ex-tech': 'Visually this is a stunning film. The special effects are seamless and the night scenes are lit ' +
    'beautifully. The score, on the other hand, is loud and generic, and the sound mix buries half of the ' +
    'dialogue. The director clearly cares more about spectacle than about rhythm, so the action scenes are ' +
    'impressive but the story in between is thin. A decent popcorn movie, nothing more.',
  'ex-short': 'One of the best films I have seen this year. Go see it.',
};

function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
const num = (x, d = 1) => x.toLocaleString('vi-VN', {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = p => num(p * 100, p >= 0.995 || p < 0.005 ? 0 : 1) + '%';

function count() {
  $('count').textContent = $('review').value.length.toLocaleString('vi-VN') + ' / 20.000 ký tự';
}
for (const [id, text] of Object.entries(SAMPLES)) $(id).onclick = () => { $('review').value = text; count(); };
$('review').oninput = count;

function aspectCard(aspect, main, others) {
  const card = el('article', undefined, 'aspect ' + main.label);
  const head = el('div', undefined, 'aspect-head');
  head.append(el('h2', ASPECT_NAMES[aspect] || aspect), el('span', ASPECT_HINTS[aspect] || '', 'hint'));
  card.append(head, el('div', LABEL_NAMES[main.label], 'verdict ' + main.label));

  const bars = el('div', undefined, 'bars');
  for (const lab of LABEL_ORDER) {
    const p = main.probabilities[lab];
    const row = el('div', undefined, 'bar-row');
    const bar = el('div', undefined, 'bar');
    const fill = el('span', undefined, lab);
    fill.style.width = (p * 100) + '%';
    bar.append(fill);
    row.append(el('span', LABEL_NAMES[lab], 'bar-label'), bar, el('span', pct(p), 'bar-value'));
    bars.append(row);
  }
  card.append(bars);

  for (const o of others) {
    const r = o.result.aspects[aspect];
    const line = el('p', undefined, 'other');
    const agree = r.label === main.label;
    line.append(
      el('span', o.name + ': '),
      el('strong', LABEL_NAMES[r.label], r.label),
      el('span', ' (' + pct(r.probabilities[r.label]) + ')' + (agree ? ' · khớp' : ' · khác'), agree ? '' : 'differs'),
    );
    card.append(line);
  }
  return card;
}

$('run').onclick = async () => {
  $('run').disabled = true;
  $('message').textContent = 'Đang phân tích…';
  $('results').replaceChildren();
  $('legend').hidden = true;
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: $('review').value}),
      signal: AbortSignal.timeout(120000),
    });
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
    const [first, ...rest] = data.models;
    const main = data.results[first.key];
    const others = rest.map(m => ({name: m.name, result: data.results[m.key]}));
    $('message').textContent =
      `Kết quả của ${first.name}. ${main.token_count.toLocaleString('vi-VN')} token` +
      (main.truncated ? ` · review dài, chỉ đọc ${main.kept_tokens} token đầu và cuối` : '') +
      ` · ${num(data.seconds)} giây trên ${data.device.toUpperCase()}`;
    for (const aspect of data.aspects) $('results').append(aspectCard(aspect, main.aspects[aspect], others));
    $('legend').hidden = false;
  } catch (error) {
    $('message').textContent = error.name === 'TimeoutError'
      ? 'Quá thời gian chờ; mô hình có thể vẫn đang xử lý. Thử lại sau.' : error.message;
  } finally {
    $('run').disabled = false;
  }
};

(async () => {
  try {
    const s = await (await fetch('/health')).json();
    $('device').textContent = '● Chạy tại máy · ' + s.device.toUpperCase();
    $('models').textContent = 'Mô hình: ' + s.models.map(m =>
      `${m.name} (seed ${m.seed}, epoch ${m.epoch}, Macro-F1 validation ${num(m.validation_macro_f1 * 100)}%)`
    ).join(' · ') + '. Seed được chọn theo validation, không nhìn tập test.';
  } catch {
    $('models').textContent = 'Chưa đọc được thông tin mô hình.';
  }
})();
