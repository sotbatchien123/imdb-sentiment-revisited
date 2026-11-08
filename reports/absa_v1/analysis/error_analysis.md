# Phân tích lỗi ABSA trên tập test

25.000 review test. Nhãn so sánh: silver (Qwen3-8B, prompt v3) — chưa phải nhãn người. Số liệu là trung bình 3 seed (42/43/44).

## 1. F1 từng lớp (RoBERTa-large)

| Khía cạnh | positive | negative | neutral | none | Tỉ lệ none (silver) |
|---|---|---|---|---|---|
| acting | 88.0 | 85.4 | 50.2 | 86.6 | 26% |
| story | 75.0 | 87.7 | 50.8 | 73.0 | 14% |
| visual | 84.2 | 77.2 | 38.8 | 92.8 | 64% |
| sound | 87.1 | 79.8 | 35.5 | 97.6 | 82% |
| direction | 79.5 | 82.2 | 41.1 | 95.4 | 76% |
| overall | 93.5 | 94.2 | 53.5 | 51.5 | 2% |

## 2. F1 lớp neutral theo mô hình

| Khía cạnh | RoBERTa-large | BERT-base | BERT frozen + BiLSTM |
|---|---|---|---|
| acting | 50.2 | 38.7 | 30.5 |
| story | 50.8 | 41.7 | 37.5 |
| visual | 38.8 | 23.5 | 17.7 |
| sound | 35.5 | 23.2 | 18.5 |
| direction | 41.1 | 29.5 | 23.1 |
| overall | 53.5 | 36.5 | 33.2 |

## 3. Kiểu nhầm phổ biến nhất (RoBERTa-large)

- **acting:** neutral→positive 12%, positive→neutral 11%, negative→none 10%
- **story:** neutral→positive 16%, negative→neutral 13%, neutral→negative 13%
- **visual:** none→negative 18%, none→positive 15%, none→neutral 10%
- **sound:** none→negative 17%, none→positive 12%, positive→none 12%
- **direction:** none→negative 17%, none→positive 16%, negative→none 13%
- **overall:** negative→neutral 21%, positive→neutral 20%, neutral→positive 17%

## 4. Theo độ dài review (Macro-F1 trung bình 6 khía cạnh)

| Mô hình | ≤ 512 token | > 512 token |
|---|---|---|
| RoBERTa-large | 75.1 | 66.8 |
| BERT-base | 68.6 | 61.2 |
| BERT frozen + BiLSTM | 63.0 | 57.2 |

## 5. Overall dự đoán theo số sao IMDb (RoBERTa-large, % dự đoán)

| Sao | Số review | positive | negative | neutral | none |
|---|---|---|---|---|---|
| 1 | 5022 | 1.4 | 95.5 | 2.1 | 0.9 |
| 2 | 2302 | 1.9 | 92.2 | 4.8 | 1.1 |
| 3 | 2541 | 2.3 | 84.6 | 11.8 | 1.4 |
| 4 | 2635 | 5.0 | 70.4 | 23.1 | 1.6 |
| 7 | 2307 | 70.1 | 4.4 | 22.6 | 2.9 |
| 8 | 2850 | 84.6 | 2.0 | 10.7 | 2.7 |
| 9 | 2344 | 89.4 | 1.0 | 6.5 | 3.1 |
| 10 | 4999 | 92.0 | 1.2 | 3.5 | 3.2 |

## 6. Ca khó: cả 9 lượt chạy cùng dự đoán một nhãn khác silver

Đây là ứng viên cho nhãn silver sai, nên ưu tiên khi nhóm kiểm tay.

| Khía cạnh | Số review | Tỉ lệ | Cặp (silver → mô hình) phổ biến |
|---|---|---|---|
| acting | 1079 | 4.3% | neutral→positive (163), negative→none (155), positive→none (136) |
| story | 1614 | 6.5% | neutral→negative (369), neutral→positive (336), negative→none (151) |
| visual | 826 | 3.3% | none→negative (171), none→positive (129), positive→none (115) |
| sound | 394 | 1.6% | positive→none (85), negative→none (77), neutral→none (58) |
| direction | 626 | 2.5% | negative→none (158), none→negative (95), positive→none (83) |
| overall | 658 | 2.6% | neutral→positive (185), neutral→negative (151), negative→neutral (76) |

### Ví dụ

- **acting** `test/neg/9954_4` (4 sao): silver **positive**, mô hình **negative**. Bằng chứng LLM: “Zucco and Strange's presence, along with the cute Anne Nagel”. Trích: “… consisting only of a beard and dimestore fangs for the most part. If it were not for Zucco and Strange's presence, along with the cute Anne Nagel, this would be completely unwatchable. Strange, who would go on to play Frankenstein's monster for Unuiversal in …”
- **acting** `test/neg/5146_3` (3 sao): silver **positive**, mô hình **negative**. Bằng chứng LLM: “the seemingly effortless performance of Lynda Bellingham hitting all the right comedy marks”. Trích: “…r, with the scenery of the Isle of Wight certainly providing a beautiful setting, and the seemingly effortless performance of Lynda Bellingham hitting all the right comedy marks. This, however is not enough. A number of performances are noticeable , especially…”
- **acting** `test/pos/9002_7` (7 sao): silver **neutral**, mô hình **none**. Bằng chứng LLM: “It's been given the Hollywood treatment to entertain people”. Trích: “…art having a breakdown about this movie (those who play rugby anyway) this is a film! It's been given the Hollywood treatment to entertain people and therefore those who play rugby (myself included) are naturally gonna pick holes in the choreography of the gam…”
- **story** `test/neg/764_2` (2 sao): silver **neutral**, mô hình **negative**. Bằng chứng LLM: “resulting in any of the then-typical dancing-skeleton-type gags”. Trích: “My God, the things that passed for entertainment in this country... This is *not* the "Tom and Jerry" you may have enjoyed on Saturday Mornings, featuring a hapless cat and a clever mouse. This is a much earlier animation series, featuring a pair of Mutt-and-J…”
- **story** `test/neg/6983_4` (4 sao): silver **neutral**, mô hình **negative**. Bằng chứng LLM: “The story deals with a US war plane which is filled with gold, money, and medals, which crashes into a lake in New Zealand during WWII.”. Trích: “…w and the action just too sparse for this New Zealand adventure to deliver the goods. The story deals with a US war plane which is filled with gold, money, and medals, which crashes into a lake in New Zealand during WWII. The plane remains lost for about forty…”
- **story** `test/pos/382_10` (10 sao): silver **neutral**, mô hình **positive**. Bằng chứng LLM: “the story is told so well that I don't even think of criticizing its plausibility”. Trích: “…utly told, with many ironic twists and visual pleasures--even if it's "unbelievable." The story is told so well that I don't even think of criticizing its plausibility (although I must confess that the tennis match seems the weakest part to me--too much Hollyw…”
- **visual** `test/neg/1297_2` (2 sao): silver **none**, mô hình **negative**. Bằng chứng LLM: “”. Trích: “This film was total rubbish. I was sitting watching this absolutely furious that this was funded. They most definitely spent all their budget on the special FX rather than the script writer and director AND some of the actors. First the acting was bland and ru…”
- **visual** `test/neg/6169_4` (4 sao): silver **none**, mô hình **negative**. Bằng chứng LLM: “”. Trích: “Let me first state that I rarely review movies, I only comment if I'm blown away or disappointed in something that I thought was going to be good. Killshot was a major disappointment on so many levels. The script was horrible, the acting was sub-par (espically…”
- **visual** `test/neg/3025_1` (1 sao): silver **neutral**, mô hình **negative**. Bằng chứng LLM: “black and white movie”. Trích: “… of this movie. As I am a hardcore Eraserhead fan, the "what, you can't watch a wierd black and white movie with little-to-no dialogue?" defense does not apply. I simply can't watch TERRIBLE weird black and white movies with little-to-no dialogue. This movie i…”
- **sound** `test/neg/6429_1` (1 sao): silver **negative**, mô hình **none**. Bằng chứng LLM: “bad music”. Trích: “…d why Sara was chosen. As the film goes on, we just watch pseudo-scary scenes, with a bad music. Moreover, the end doesn't tell anything : we just see two possibilities as if one of them was a bonus scene or the director's cut... There's no plot, so no possibl…”
- **sound** `test/pos/8057_9` (9 sao): silver **positive**, mô hình **none**. Bằng chứng LLM: “Music by Harry Sukman contributes to the film effectively”. Trích: “…er are in B/W and are very much like news-film, adding to the film's realistic power. Music by Harry Sukman contributes to the film effectively. Writer Daniel Mainwaring and Crane Wilbur produced a swift-paced and straightforward story that divides into parts.…”
- **sound** `test/neg/6224_1` (1 sao): silver **none**, mô hình **negative**. Bằng chứng LLM: “”. Trích: “There's no better way to describe the HORRENDOUS experience that's to watch Mamma Mia: Hitler must be looking up, saying to the pseudo-director Phyllida Lloyd 'You monster, what have you done?' Everything about the movie is wrong. Just Wrong! Even its success,…”
- **direction** `test/neg/407_4` (4 sao): silver **neutral**, mô hình **negative**. Bằng chứng LLM: “You've got to feel for director Brett Piper as he clearly didn't have much to work with for this film”. Trích: “…ng even resembling a third dimension and they will soon begin to thoroughly bore you. You've got to feel for director Brett Piper as he clearly didn't have much to work with for this film; but that doesn't make Arachnia worth a damn, and overall there are bett…”
- **direction** `test/pos/6896_10` (10 sao): silver **neutral**, mô hình **none**. Bằng chứng LLM: “The movie is told by the younger of the two as he is explaining the business to a priest before he is sent to be beheaded.”. Trích: “…ent clientèle they can serve. This changes the direction of the corpses they collect. The movie is told by the younger of the two as he is explaining the business to a priest before he is sent to be beheaded. His partner had already been beheaded. The priest i…”
- **direction** `test/pos/11442_10` (10 sao): silver **neutral**, mô hình **positive**. Bằng chứng LLM: “Director Frank D. Gilroy also wrote the script”. Trích: “…s success. This is a little slice-of-life movie that is most entertaining throughout. Director Frank D. Gilroy also wrote the script and it's full of interesting subplots and unexpected twists. The actors are journeymen who do a solid job. The biggest revelati…”
- **overall** `test/neg/11143_2` (2 sao): silver **negative**, mô hình **neutral**. Bằng chứng LLM: “There probably isn't a worse way to spend a Saturday night”. Trích: “…ation. But they were probably desperate to get in front of Morrissey's camera anyway. There probably isn't a worse way to spend a Saturday night but at least it brings a specific time and place vividly to life.”
- **overall** `test/neg/9587_2` (2 sao): silver **neutral**, mô hình **negative**. Bằng chứng LLM: “It sounds like the BBC have decided to turn it into a more conventional 30 minute sitcom”. Trích: “…ked it ! There will be a second series in Autumn 2007 - though without the male lead. It sounds like the BBC have decided to turn it into a more conventional 30 minute sitcom.”
- **overall** `test/neg/3185_1` (1 sao): silver **neutral**, mô hình **negative**. Bằng chứng LLM: “It was okay”. Trích: “…e efforts of the actors. I probably would have gone out of the theatre thinking, "Eh. It was okay." So why the undeniably hostile tone in my review? The ending. If, as it's been noted, the rest of the movie is just all a setup for the ending, then it misses sp…”
