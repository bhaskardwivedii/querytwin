# Error analysis — ens on 40,429 held-out test pairs

## Operating points

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| default (0.50) | 0.8510 | 0.8020 | 0.7918 | 0.7969 |
| F1-tuned on validation (0.42) | 0.8485 | 0.7628 | 0.8556 | 0.8065 |

Lowering the threshold trades precision for recall: use it when missing a duplicate costs more than a false merge (e.g. suggesting "similar questions" to a user).

## Accuracy by question length (words in the longer question)

| Length | Pairs | Accuracy |
|---|---|---|
| 0–7 | 5,970 | 0.8307 |
| 8–11 | 15,629 | 0.8191 |
| 12–17 | 11,706 | 0.8581 |
| 18–29 | 6,224 | 0.9198 |
| 30+ | 900 | 0.9711 |

## Most confident false positives (predicted duplicate, labelled different)

| p(dup) | Question 1 | Question 2 |
|---|---|---|
| 1.000 | What does my birth chart say about me? | What does my birth chart say about me?  |
| 1.000 | What is the output of this program? ? | What is the output of this program? |
| 1.000 | How would you rephrase this sentence?  | How would you rephrase this sentence? |
| 1.000 | If you were a farmer what would you grow? | If you were a farmer, what would you grow? |
| 0.982 | Was Hillary Clinton responsible for Benghazi? | Did Hillary Clinton cause Benghazi? |
| 0.981 | How do I get free traffic for my website? | How do you get good free traffic to your website? |
| 0.980 | What's your favourite joke? | What is your favorite joke? |
| 0.979 | What will you do with your black money after the government discontinued 500 and 1000 notes? | What do you think of the decision by the Indian Government to demonetize 500 and 1000 rupee notes? |
| 0.976 | How do I get more money online? | How do I get more money? Is there a way online? |
| 0.972 | How do I get more traffic on my blog website? | How can I increase the traffic on my website? |
| 0.972 | What are the roles of men and women in when rain clouds gather? | Roles of men and women in when rain clouds gather? |
| 0.972 | Is keeping a old Rs. 500 or Rs. 1000 note as a memory is a crime? | What were the security features of an old Rs. 500 and Rs. 1000 currency notes? |

## Most confident false negatives (predicted different, labelled duplicate)

| p(dup) | Question 1 | Question 2 |
|---|---|---|
| 0.002 | Where can I sell a B2B list that lets me email specific contacts at target prospect companies, and followup with a phone call to that contact? | Where can I buy a B2B list that lets me email specific contacts at target prospect companies, and followup with a phone call to that contact? |
| 0.008 | I am afraid I am the only conscious person and everyone else is just a deterministic natural robot. Is there an irrefutable argument that would disprove this fear despite its unlikely nature? | How do you prove that I am not the only person in the Matrix and everyone else is just a computer program? |
| 0.008 | Was Karl Marx a genius even if he was wrong on the big picture? | Is Karl Marx overrated? |
| 0.009 | What are the advantages of intensive cultivation? | What are the disadvantages of intensive cultivation? |
| 0.009 | What is the meaning of the Hindi word "Gaandu” (slang)? | What is the meaning of word "gandoo"? |
| 0.010 | How does Docking and berthing of spacecraft work? | What is the difference between docking and berthing at the ISS? |
| 0.011 | What was going on in the Columbia Space Shuttle just before its explosion? | Did the Columbia crew in the shuttle know that they were going to die during the last fifteen minutes? |
| 0.012 | What are some good short term courses for an aspiring Chartered Accountant which enhances their job prospects? | What are some good short term courses that I can do along with CA so that it benefits me and gets reflected on my C.V.? |
| 0.013 | What are the causes of NPA? | Why are Indian banks so overburdened by the NPA? |
| 0.014 | Is there any problem during certificate verification in govt. jobs if there is only one letter spelling mistake in 10th and 12th certificate? | My name in class 10th certificate is X Y Z whereas in class 12th certificate it is X Y. Will it cause any problem during admission in colleges? |
| 0.015 | Why do Orthodox Jewish men wear black hats (not referring to streimels)? | What are the origins of black hats and attire worn by Orthodox Jews? |
| 0.015 | Is prostitution in Saudi Arabia illegal? | Is prostitution legal in Saudi Arabia? |

Some of these are label noise: Quora's labels are human judgements and the dataset authors note they are not guaranteed to be perfect.
