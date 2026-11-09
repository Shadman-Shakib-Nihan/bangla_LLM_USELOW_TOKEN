# Phase 4b: M4 multi-sentence examples

## Example 1

- Dataset: `tydi_heldout`
- Retriever: `Dense (e5-small)`
- Question ID: `tydi_bengali-7478984229355295611-0`
- Question: জাতিসংঘ শান্তিরক্ষা মিশন প্রথম কোথায় শুরু হয় ?
- Gold answer(s): মধ্যপ্রাচ্যে
- Selected unique sentences: 2
- o200k tokens: 86
- e5 tokens: 73

```text
এর প্রথম মিশন ছিল ১৯৪৮ সালে মধ্যপ্রাচ্যে ১৯৪৮ আরব-ইসরায়েলি যুদ্ধের সময় যুদ্ধবিরতি পালন ও বজায় রাখা।

প্রথমে শান্তিরক্ষা মিশন আরব-ইসরায়েলি যুদ্ধের সময় ১৯৪৮ সালে মধ্যপ্রাচ্যে নিযুক্ত জাতিসঙ্ঘের পর্যবেক্ষকদের একটি দল ছিল।
```

## Example 2

- Dataset: `tydi_heldout`
- Retriever: `Dense (e5-small)`
- Question ID: `tydi_bengali-3176128016413867974-0`
- Question: গায়িকা ভেনিসা অ্যানা হাডজেন্সের দ্বিতীয় অ্যালবামটির নাম কী ?
- Gold answer(s): আইডেন্টফিাইড
- Selected unique sentences: 2
- o200k tokens: 71
- e5 tokens: 56

```text
[2] তার দ্বিতীয় অ্যালবাম, আইডেন্টফিাইড, জুলাই১, ২০০৮ সালে মার্কিন যুক্তরাষ্ট্রে মুক্তি পায়।

তার দ্বিতীয় অ্যালবাম, "আইডেন্টফিাইড", জুলাই ১, ২০০৮ সালে মার্কিন যুক্তরাষ্ট্রে মুক্তি পায়।
```

## Example 3

- Dataset: `tydi_heldout`
- Retriever: `Hybrid (RRF)`
- Question ID: `tydi_bengali-7478984229355295611-0`
- Question: জাতিসংঘ শান্তিরক্ষা মিশন প্রথম কোথায় শুরু হয় ?
- Gold answer(s): মধ্যপ্রাচ্যে
- Selected unique sentences: 2
- o200k tokens: 86
- e5 tokens: 73

```text
এর প্রথম মিশন ছিল ১৯৪৮ সালে মধ্যপ্রাচ্যে ১৯৪৮ আরব-ইসরায়েলি যুদ্ধের সময় যুদ্ধবিরতি পালন ও বজায় রাখা।

প্রথমে শান্তিরক্ষা মিশন আরব-ইসরায়েলি যুদ্ধের সময় ১৯৪৮ সালে মধ্যপ্রাচ্যে নিযুক্ত জাতিসঙ্ঘের পর্যবেক্ষকদের একটি দল ছিল।
```

## Example 4

- Dataset: `tydi_heldout`
- Retriever: `Hybrid (RRF)`
- Question ID: `tydi_bengali-3176128016413867974-0`
- Question: গায়িকা ভেনিসা অ্যানা হাডজেন্সের দ্বিতীয় অ্যালবামটির নাম কী ?
- Gold answer(s): আইডেন্টফিাইড
- Selected unique sentences: 2
- o200k tokens: 71
- e5 tokens: 56

```text
তার দ্বিতীয় অ্যালবাম, "আইডেন্টফিাইড", জুলাই ১, ২০০৮ সালে মার্কিন যুক্তরাষ্ট্রে মুক্তি পায়।

[2] তার দ্বিতীয় অ্যালবাম, আইডেন্টফিাইড, জুলাই১, ২০০৮ সালে মার্কিন যুক্তরাষ্ট্রে মুক্তি পায়।
```

## Example 5

- Dataset: `tydi_validation`
- Retriever: `Dense (e5-small)`
- Question ID: `tydi_bengali--6695334746728483406-2`
- Question: ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর প্রথম প্রকাশিত উপন্যাসের নাম কী ?
- Gold answer(s): ঝাঁসির রানি, ঝাঁসির রানি
- Selected unique sentences: 2
- o200k tokens: 73
- e5 tokens: 66

```text
[9] তাঁর প্রথম উপন্যাস ঝাঁসির রানি ঝাঁসির রানির (লক্ষ্মীবাই) জীবনী অবলম্বনে রচিত।

তাঁর প্রথম উপন্যাস "ঝাঁসির রানি" ঝাঁসির রানির (লক্ষ্মীবাই) জীবনী অবলম্বনে রচিত।
```

