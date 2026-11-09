# Phase 4: five verbatim M2 answer-loss examples

M2 uses e5-small sentence similarity. Each example is a TyDi question where the M0 top-10 context contained a gold answer but the M2 context did not. Text below is copied verbatim from the dataset and the experiment output.

## Example 1

- Dataset: `tydi_heldout`
- Retriever: `Dense (e5-small)`
- Budget: 128 o200k_base tokens
- Question ID: `tydi_bengali--1146659921200539894-2`

**Question**

> জ্যোতি বসুর আদি বাড়ি কোথায় ?

**Gold answer(s)**

> ঢাকা জেলার (অধুনা বাংলাদেশ রাষ্ট্রের নারায়ণগঞ্জ জেলার সোনারগাঁ উপজেলার বারদী গ্রামে

**Original answer-bearing sentence (verbatim)**

> উল্লেখ্য, বসু পরিবারের আদিনিবাস ছিল ব্রিটিশ বাংলা প্রদেশের ঢাকা জেলার (অধুনা বাংলাদেশ রাষ্ট্রের নারায়ণগঞ্জ জেলার সোনারগাঁ উপজেলার বারদী গ্রামে।

Parent passage: `tydi_648beb2eeaf4b4f9` at retrieval rank 1.

**M2 compressed context (verbatim output)**

```text
ডা: নিশিকান্ত বসু ও হেমলতা বসুর তৃতীয় সন্তান জ্যোতি বসুর জন্ম ১৯১৪ খ্রিস্টাব্দের ৮ জুলাই কলকাতার ৪৩/১ হ্যারিসন রোডস্থ (বর্তমান মহাত্মা গান্ধী রোড) বাসভবনে।

জ্যোতির বাড়ি ময়মনসিংহের গৌরীপুরে।

তার পৈতৃক বাড়ি ছিল পাবনা জেলায়।

তিনি ছিলেন মনিষী রাজনারায়ণ বসুর ভ্রাতুষ্পুত্র।
```

**Why the answer was lost**

None of the 2 answer-bearing sentence(s) made the budget cut. The best one ranked 13 by e5 similarity (0.810) and needed 46 o200k tokens; higher-scoring sentences filled the 128-token budget.

## Example 2

- Dataset: `tydi_heldout`
- Retriever: `Dense (e5-small)`
- Budget: 256 o200k_base tokens
- Question ID: `tydi_bengali--1450501750137438198-0`

**Question**

> শ্রীশ্রীঠাকুর অনুকূলচন্দ্র কোথায় জন্মগ্রহণ করেন ?

**Gold answer(s)**

> ব্রিটিশ ভারতের বঙ্গ প্রদেশের পাবনা জেলার হিমায়তপুরে

**Original answer-bearing sentence (verbatim)**

> তিনি ব্রিটিশ ভারতের বঙ্গ প্রদেশের পাবনা জেলার হিমায়তপুরে জন্মগ্রহণ করেন, যা বর্তমানে বাংলাদেশের অন্তর্গত।

Parent passage: `tydi_80f7d5c0cdca2075` at retrieval rank 1.

**M2 compressed context (verbatim output)**

```text
শ্রীশ্রীঠাকুর অনুকূলচন্দ্র (১৪ই সেপ্টেম্বর, ১৮৮৮- ২৬শে জানুয়ারি, ১৯৬৯) বাঙালি ধর্ম সংস্কারক।

ঈশ্বরচন্দ্র বিদ্যাসাগর ১৮২০ খ্রিস্টাব্দের ২৬ সেপ্টেম্বর (বাংলা ১২২৭ বঙ্গাব্দের ১২ আশ্বিন, মঙ্গলবার) বর্তমান পশ্চিম মেদিনীপুর জেলার বীরসিংহ গ্রামে জন্মগ্রহণ করেন।

হরিশ্চন্দ্রের জন্ম হয় বর্তমান কলকাতা শহরের ভবানীপুরে ।

ঈশ্বরচন্দ্র বিদ্যাসাগর ১৮২০ খ্রিস্টাব্দের ২৬ সেপ্টেম্বর (বাংলা ১২২৭ বঙ্গাব্দের ১২ আশ্বিন, মঙ্গলবার) বর্তমান পশ্চিম মেদিনীপুর জেলার বীরসিংহ গ্রামে জন্মগ্রহণ করেন।

শাস্ত্রীয় বিবরণ ও জ্যোতিষ গণনার ভিত্তিতে লোক বিশ্বাস অনুযায়ী ভগবান শ্রীকৃষ্ণের জন্ম হয়েছিল ৩২২৮ খ্রীষ্টপূর্বাব্দের ১৮ অথবা ২১ জুলাই। এ দিনটি জন্মাষ্টমী নামে পরিচিত।
```

**Why the answer was lost**

None of the 1 answer-bearing sentence(s) made the budget cut. The best one ranked 15 by e5 similarity (0.796) and needed 31 o200k tokens; higher-scoring sentences filled the 256-token budget.

## Example 3

- Dataset: `tydi_heldout`
- Retriever: `Hybrid (RRF)`
- Budget: 128 o200k_base tokens
- Question ID: `tydi_bengali--1630283428092480500-8`

**Question**

> লেসবিয়ান শব্দের বাংলা পরিভাষা কী ?

**Gold answer(s)**

> নারী সমকামী

**Original answer-bearing sentence (verbatim)**

> গে শব্দটির দ্বারা পুরুষ সমকামীদের বোঝানো হয় এবং নারী সমকামীদেরকে বোঝানো হয় লেসবিয়ান শব্দটির দ্বারা।

Parent passage: `tydi_274f432c6d158b70` at retrieval rank 1.

**M2 compressed context (verbatim output)**

```text
আর ‘লেসবিয়ান’ শব্দটি এসেছে গ্রিস দেশের ‘লেসবো’ নামক দ্বীপমালা থেকে।

আর ‘লেসবিয়ান’ শব্দটি এসেছে গ্রিস দেশের ‘লেসবো’ নামক দ্বীপমালা থেকে।

এর অর্থ ব্যাপক।

পুরুষালি ধরনের পোশাকপরিচ্ছদ বাচ লেসবিয়ানদের স্টাইল।

লেসবিয়ান মাত্রই "বাচ" বা পুরুষালি পোশাকের নারী এই ধারণাকে ভেঙে শোটিতে দেখানো হয় "মেয়েলি" বা "লিপস্টিক" লেসবিয়ান।
```

**Why the answer was lost**

None of the 2 answer-bearing sentence(s) made the budget cut. The best one ranked 6 by e5 similarity (0.838) and needed 34 o200k tokens; higher-scoring sentences filled the 128-token budget.

## Example 4

- Dataset: `tydi_heldout`
- Retriever: `Hybrid (RRF)`
- Budget: 256 o200k_base tokens
- Question ID: `tydi_bengali--1757779693302249107-0`

**Question**

> কেনিয়া কত সালে পূর্ব আফ্রিকার একটি প্রজাতান্ত্রিক রাষ্ট্র হিসেবে পরিচিতি লাভ করে ?

**Gold answer(s)**

> ১৯৬৪

**Original answer-bearing sentence (verbatim)**

> ১৯৬৩ সালে এটি স্বাধীনতা লাভ করে এবং ১৯৬৪ সাল থেকে এটি একটি প্রজাতান্ত্রিক রাষ্ট্র হয়।

Parent passage: `3134#0` at retrieval rank 1.

**M2 compressed context (verbatim output)**

```text
কেনিয়া পূর্ব আফ্রিকার একটি প্রজাতান্ত্রিক রাষ্ট্র এবং কমনওয়েলথ অভ নেশনসের সদস্য।

কেনিয়া পূর্ব আফ্রিকার একটি প্রজাতান্ত্রিক রাষ্ট্র এবং কমনওয়েলথ অভ নেশনসের সদস্য।

দ্য রিপাবলিক অব কেনিয়া পূর্ব আফ্রিকার একটি দেশ। আফ্রিকার দ্বিতীয় সর্বোচ্চ পর্বতশিখর ‘মাউন্ড কেনিয়া’ থেকেই কেনিয়া নামটি নেয়া হয়েছে। প্রাগৈতিহাসিক যুগে ‘মাউন্ট কেনিয়া’র অগ্নুৎপাতের ফলে কেনিয়া সৃষ্টি হয়।

কেনিয়া পর্বতের নাম থেকেইকেনিয়া প্রজাতন্ত্রের নামটি এসেছে।

১৪৯৮ সালে পর্তুগিজদের আগমনের সাথে কেনিয়ার উপকূলে আরব আধিপত্যের অবসান ঘটে এবং প্রায় ১৫০ বছর ধরে পর্তুগিজরা এখানে আধিপত্য বজায় রাখে। ১৯শ শতকের মধ্যভাগে পূর্ব আফ্রিকাতে ব্রিটিশরা অভিযান চালায় এবং ১৮৯৫ সালে কেনিয়া একটি ব্রিটিশ-রক্ষিত রাষ্ট্রে বা প্রোটেক্টোরেটে পরিণত হয়।
```

**Why the answer was lost**

None of the 3 answer-bearing sentence(s) made the budget cut. The best one ranked 12 by e5 similarity (0.813) and needed 26 o200k tokens; higher-scoring sentences filled the 256-token budget.

## Example 5

- Dataset: `tydi_validation`
- Retriever: `Dense (e5-small)`
- Budget: 128 o200k_base tokens
- Question ID: `tydi_bengali--1366347128155871013-24`

**Question**

> হিন্দু শাস্ত্র অনুযায়ী গণেশের সহোদর কে ?

**Gold answer(s)**

> কার্তিকেয়

**Original answer-bearing sentence (verbatim)**

> অবতারকাল সমাপ্ত হলে ময়ূরটি তিনি তাঁর ভ্রাতা কার্তিকেয়কে দান করেন।

Parent passage: `tydi_200abe9fbe65d22b` at retrieval rank 6.

**M2 compressed context (verbatim output)**

```text
হিন্দু পুরাণের নিয়ম অনুসারে, অন্যান্য দেবতার আগে গণেশের পূজা করতে হয়।

গণেশ সহস্রনাম (; ) হল হিন্দু দেবতা গণেশের নাম-সংকীর্তন স্তোত্র।

খ্রিস্টীয় ৯ম শতাব্দীতে হিন্দুধর্মের অন্যতম শাখা স্মার্ত সম্প্রদায়ের পাঁচ জন প্রধান দেবতার তালিকায় গণেশের নাম অন্তর্ভুক্ত করা হয়।

ত্রেতাযুগের গণেশ ‘ময়ূরেশ্বর’ রূপধারী।

বাহন ময়ূর।
```

**Why the answer was lost**

None of the 1 answer-bearing sentence(s) made the budget cut. The best one ranked 69 by e5 similarity (0.763) and needed 26 o200k tokens; higher-scoring sentences filled the 128-token budget.

