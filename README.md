## Week 7: Text and Multimodal AI

- **Text dataset:** SMS Spam Collection (UCI/Kaggle), 5,572 messages (4,825 ham, 747 spam); stratified 80/20 split, test set 1,115 messages.
- **Word n-gram classifier (1-2 grams, 12,128 features):** macro-F1 = **0.974**, spam precision = 0.979, spam recall = 0.933.
- **Character n-gram classifier (2-5 grams, 36,449 features):** macro-F1 = **0.982**, spam precision = 0.986, spam recall = 0.953.
- **Feature analysis:** strongest spam-associated features = call, txt, free, text, reply; strongest ham-associated features = my, me, ok, that, it. Some (uk, to, your) are shortcuts rather than real spam language.
- **Tricky messages:** my Roman Urdu spam message scored only 0.44 (word) and 0.23 (char), so it would be missed.
- **NER:** provided gold-set F1 = **0.533** before and **0.875** after the entity ruler. My EuroSAT domain held-out F1 = **0.467** before and **0.400** after rules (first five: 0.480 to 0.923, inflated because the rules were written from those sentences).
- **Regex:** extracted satellite tile IDs, acquisition dates, coordinates, area, cloud percentage and EuroSAT land-cover labels; **9/9** positive and negative tests passed, and `\b` was shown to prevent over-matching.
- **OCR:** controlled pairs gave CER 0.062, 0.059 and 0.143 down to **0.000** after context-based correction; EasyOCR on the generated survey report and on the lab invoice gave CER **0.000** raw and corrected (clean synthetic images are easy).
- **Multimodal pipeline:** Week 6 ResNet18 probabilities + a masked survey note produce an explainable JSON case record with a human-review decision. Phone and email masking was verified before NER and regex extraction.
- **Limitations:** small NER gold set (10 sentences), synthetic OCR samples, European EuroSAT imagery with possible geographic and seasonal shift, and no automatic legal or planning decisions.

## Responsibility
The system provides evidence for human review. It must not make legal land-title, tax, subsidy or zoning decisions automatically. Geographic validation and independent review are required before operational use.
