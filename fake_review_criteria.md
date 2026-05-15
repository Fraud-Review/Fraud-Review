# Fake Review Labeling Criteria

The labels are model judgments, not ground truth. Use them as a screening result
that should be spot-checked before making claims.

## real

- Review contains concrete gameplay experience, mechanics, context, pros/cons, or
  a plausible personal reaction.
- Account/playtime metadata is broadly consistent with the review.
- Short reviews can still be real when metadata is strong and the text looks like
  normal human feedback.

## suspicious

- Review is extremely generic, duplicated-looking, meme-only, or too short to
  judge confidently.
- Metadata has weak or mixed signals, such as very low playtime, one total review,
  free copy, or many similar low-effort reviews.
- Text may be sincere but lacks enough evidence for a confident real/fake label.

## fake

- Review looks like spam, advertising, review farming, copy-paste manipulation,
  bot output, irrelevant text, or coordinated praise/attack.
- Strong mismatch between text and metadata, such as confident claims with almost
  no playtime, or repeated boilerplate from low-history accounts.
- Text is mostly links, scams, commands, unrelated content, or unnatural keyword
  stuffing.

The script passes both review text and metadata to the model. The final percentage
is calculated from the model's JSON labels in the output CSV.
