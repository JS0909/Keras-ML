from transformers import pipeline

# classifier = pipeline('sentiment-analysis')
# res = classifier('I like reading a book.')
# print(res)

generator = pipeline('text-generation', model='distilgpt2')
res = generator(
    'I would like to',
    max_length=30,
    num_return_sequences=2
    )
print(res)

# generator = pipeline('translation_en_to_de')
# res = generator(
#     'Do you like pineapple?'
# )
# print(res)