## approach to completing the assignment

1. In the backend, I used pretrained `DistilBert` model to generate the output. I tried several other pretrained models such as `RoBERTa`, `ALBERT`, `bert-large-uncased-whole-word-masking-finetuned-squad` and among these DistilBert performed well. I tried using API too, but not yielding satisfying result as expected.
2. The files from front end was saved in a temporary folders first and then saved into `MongoDB`. While trying to save directly to the database, I faced few errors, so considered this approach.
3. After extracting the content from the temporary folder, I saved it in the database and then did tokenisation fed into the model.
4. With my limited knowledge in react, I managed to modify the frontend file.