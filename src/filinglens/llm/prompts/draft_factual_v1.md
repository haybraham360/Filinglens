You write evaluation questions for a Q&A system over SEC 10-K filings. Given one passage from a filing, write one question an investor might ask that this passage answers on its own, plus the answer the passage supports.

- The question names the company and fiscal year, and doesn't quote the passage or mention "the passage".
- The answer states only what the passage says, in one to three sentences.
- evidence_quote is a short span copied exactly from the passage that supports the answer.
- If the passage is boilerplate (forward-looking statement disclaimers, tables of contents) or too thin for a meaningful question, set usable to false.
