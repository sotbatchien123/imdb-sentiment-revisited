const $=id=>document.getElementById(id);
const names={bilstm:'Chỉ đọc phần đầu',bilstm_headtail:'Đọc phần đầu + cuối',multichunk:'Đọc toàn bộ bằng nhiều đoạn'};
const samples={positive:'The performances are wonderful and the story is deeply moving. I enjoyed every minute of this beautifully directed film. Highly recommended.',negative:'The plot makes no sense and the acting is painfully wooden. I was bored throughout the movie. This was a complete waste of time.',mixed:('At first the film looks promising. The setting is attractive and the opening music creates a warm atmosphere. The characters travel from one town to another, talking about their plans and meeting new people along the way. ').repeat(8)+'However, none of these scenes leads anywhere. The dialogue becomes repetitive and the ending resolves nothing. Despite the beautiful opening, I found the film frustrating and would not recommend it.'};
function count(){$('count').textContent=$('review').value.length.toLocaleString('vi-VN')+' / 20.000 ký tự';}
for(const key of Object.keys(samples))$(key).onclick=()=>{$('review').value=samples[key];count();};
$('review').oninput=count;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
$('run').onclick=async()=>{
  $('run').disabled=true;$('message').textContent='Đang phân tích trên CPU…';$('results').replaceChildren();
  try{const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('review').value}),signal:AbortSignal.timeout(120000)});const data=await response.json();if(!response.ok)throw Error(data.error);
    $('message').textContent=`${data.token_count} token (gồm token đặc biệt) · ${data.long?'Review dài':'Review ngắn'} · ${data.seconds.toFixed(2)} giây trên CPU`;
    for(const [kind,p] of Object.entries(data.predictions)){
      const card=el('article',undefined,'card'),selection=data.selection[kind],visible=data.visible_input[kind];card.append(el('h2',names[kind]),el('p',`Checkpoint chọn bằng validation · seed ${selection.seed} · epoch ${selection.epoch}`,'hint'),el('div',p.label?'Tích cực':'Tiêu cực','verdict '+(p.label?'positive':'negative')));
      const bar=el('div',undefined,'bar'),fill=el('span');fill.style.width=(p.positive_probability*100)+'%';bar.append(fill);card.append(bar,el('p',`P(tích cực): ${(p.positive_probability*100).toFixed(1)}% · P(tiêu cực): ${((1-p.positive_probability)*100).toFixed(1)}%`,'prob'),el('div',`Giữ ${visible.retained_content_tokens} token nội dung · Bỏ ${visible.discarded_content_tokens} token`,'tokens'));
      const details=el('details');details.append(el('summary','Xem văn bản mô hình đã đọc'));visible.segments.forEach((text,i)=>{if(i)details.append(el('p',kind==='bilstm_headtail'?'… phần giữa đã được bỏ …':'Đoạn tiếp theo','hint'));details.append(el('div',text,'segment'));});card.append(details);$('results').append(card);
    }
  }catch(error){$('message').textContent=error.name==='TimeoutError'?'Yêu cầu quá thời gian; mô hình có thể vẫn đang xử lý. Hãy thử lại sau.':error.message;}
  finally{$('run').disabled=false;}
};
async function status(){try{const r=await fetch('/api/status');const s=await r.json();const stage={training:'Đang huấn luyện',validation:'Đang đánh giá validation',completed:'Đã hoàn tất 3 seed',completed_pilot_gate_not_passed:'Pilot đã xong; chưa đạt điều kiện mở rộng',failed:'Thí nghiệm gặp lỗi',paused:'Đã tạm dừng',loading:'Đang nạp mô hình'};const stale=s.updated_at&&(Date.now()-Date.parse(s.updated_at)>180000);$('training').textContent=(stage[s.stage]||s.stage)+(s.epoch?` · seed ${s.seed} · epoch ${s.epoch}/50`:'');$('training-detail').textContent=(s.completed_reviews?`${s.completed_reviews.toLocaleString('vi-VN')}/${s.total_reviews.toLocaleString('vi-VN')} review trong epoch. `:'')+(stale?'Bản ghi đã cũ; cần kiểm tra tiến trình. ':'')+(s.updated_at?'Cập nhật: '+s.updated_at:'');}catch{$('training').textContent='Chưa đọc được trạng thái.';}}
status();setInterval(status,15000);
