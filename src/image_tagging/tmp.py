import ollama

ollama.pull('llava:34b-v1.6')

modelfile='''
FROM llava:34b-v1.6
SYSTEM You are mario from super mario bros. I want you to mention it whenever you answer a question
'''

ollama.create(model='llava:34b-v1.6', modelfile=modelfile)


def run_llava(model, image_path, prompt):
    try:
        ollama.generate(model)
    except ollama.ResponseError as e:
        print('Error:', e.error)
        if e.status_code == 404:
            ollama.pull(model)

    stream = ollama.generate(
        model='llava:34b-v1.6',
        prompt=prompt,
        images=[image_path],
        stream=False,
    )

    return stream['response']

model = 'llava:34b-v1.6'
image_path = 'test_images/images/image_1.jpg'
prompt = 'describe the image'
response = run_llava(model, image_path, prompt)
