# from INIT.py import *
from sentence_clustering import *
import re
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


from flask import Flask, request  # Imports
from io import BytesIO
from google.cloud import storage
from flask_cors import CORS
import random
import string
from replit import db  # Imports Part 2
import os

data = []
label = []
with open('labels', 'rb') as fp:
    label = pickle.load(fp)

with open('sentences', 'rb') as fp:
    data = pickle.load(fp)

clusters = defaultdict(list)
for i in range(len(data)):
  clusters[label[i]].append(data[i])

res = []
for i in range(4):
    sen = []
    c = 0
    while c < 6:
      index = random.randint(0, len(clusters[c])-1)
      sentence = clusters[c][index]
      words = tokenize.word_tokenize(sentence)
      pos_tagged = nltk.pos_tag(words)

      if c == 0:
        n = list(filter(lambda x:x[1]=='JJ',pos_tagged))
      elif c == 1:
        n = list(filter(lambda x:x[1]=='VB',pos_tagged))   
      else:
        n = list(filter(lambda x:x[1]=='NN',pos_tagged))

      if len(n) != 0:
          choice = (random.choice(n))[0]
          if len(choice) > 2 and "." not in choice:
            sen.append(choice)
            c += 1
    res.append(sen)
print(res)
ans_choices = []
for r in res:
    keyword = 'Keywords: '
    for i in range(5):
      keyword = keyword + r[i] + ', '
    keyword += r[5]
    default_promt = '\nThis program will generate an introductory paragraph to a blog post given a blog title, audience, and tone of voice.\n--\nKeywords: thanos, elon musk, pixar, hairstyle\nPrompt: a full character portrait of elon musk as thanos, the pixar adaptation, with same hairstyle, hyper detailed, digital art, trending in artstation, cinematic lighting, studio quality, smooth render, unreal engine 5 rendered, octane rendered.\n--\nKeywords: monkey, business, suit.\nPrompt: Three monkeys in business suits.\n--\nKeywords: island, lake, birds, planet, miniature\nPrompt: gediminas pranckevicius fish eye view of detailed portrait of floating miniature desert island with oasis planet, intricate complexity, by greg rutkowski, ross tran, conrad roset, takato yomamoto, ilya kuvshinov palm trees, small lake, birds, rocks, painting by lucian freud and mark brooks, bruce pennington, bright colors, neon, life, god, godrays, sinister hall smoke \n--\nKeywords: swans, beautiful, lake, water\nPrompt: photo of two black swans touching heads in a beautiful reflective mountain lake, a colorful hot air balloon is flying above reflecting off water, hot air balloon, intricate, 8k highly professionally detailed, centered, HDR, CGsociety\n--\nKeywords: machines, steampunk, library\nPrompt: A digital illustration of a steampunk library with clockwork machines, 4k, detailed, trending in artstation, fantasy vivid colors\n--\nKeywords: tornado, purple, mountain, beautiful\nPrompt: amazing landscape photo of a purple tornado descending on a mountain by marc adamus, beautiful dramatic lighting\n--\nKeywords: city, plants, clouds, ground, machines\nPrompt: A highly detailed crisp unreal engine render of aerial drone photo of A beautiful futuristic cyberpunk abandoned city building with neon, plants, perfect well made rainbow on the sky, sunlight breaking through clouds, debris on the ground, abandoned machines bright warm colors by wangchen-cg, Neil blevins, artstation, Isometric japanese city, volumetrics, 3d render, octane render, Gediminas Pranckevicius\n--\nKeywords: Kitten, pizza\nPrompt: Kitten with armor made of pizza, pen and ink, intricate line drawings, by Yoshitaka Amano, Ruan Jia, Kentaro Miura, Artgerm, detailed, trending on artstation, hd, masterpiece\n--\nKeywords: blue, octopus, bottle, light, ocean\nPrompt: A bottle with a blue octopus inside, a blue light, ocean, beautiful\n--\nKeywords: lighted, think, way, worry, bother, door\nPrompt: A lighted doorway to a thinker\'s way, worry, bother, door\n--\n'
    prompt = default_promt + keyword
    response = co.generate( 
      model='xlarge', 
      prompt= prompt,
      k=0, 
      p=0.75, 
      frequency_penalty=0, 
      presence_penalty=0, 
      stop_sequences=[], 
      return_likelihoods='NONE')

    ans_choices.append(re.split(r'Prompt:|--',response.generations[0].text)[1])

    
answer = random.randint(0, 3) 
prompt = ans_choices[answer]

# print("Keywords are: " + ' '.join(res[answer]))
# print("Generated Prompt: " + prompt)
for i in range(4):
    print("Prompt " + str(i) + " is " + ans_choices[i])







model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

sample_num = 1

# prompt_input = "A lighted doorway to a thinker's way, worry, bother, door"
prompt_input = prompt
for i in range(sample_num):
    with autocast("cuda"):
        a = pipe(prompt, guidance_scale=7.5, height=512, width=512,
                 num_inference_steps=50, seed='random', scheduler='LMSDiscreteScheduler')["sample"][0]
        # display(a)
        print(type(a))
        a.save(f'outputs/gen-image-{i}.png')


# def file_upload(path):
#     storage_client = storage.Client.from_service_account_json("calhacks-365701-9935a4720153.json", project='CalHacks')
#     bucket = storage_client.get_bucket("calhacks-bucket")
#     filename = "%s%s" % ('', path)
#     blob = bucket.blob("you.png")
#     blob.content_type = "image/jpeg"
#     with open(path, 'rb') as f:
#         blob.upload_from_file(f)
#         print("Success")
        
# file_upload('/notebooks/outputs/gen-image-0.png')


# #server        
# app = Flask('app')  # Create our app
# CORS(app)

# def get_random_string(length):
#     # choose from all lowercase letter
#   letters = string.ascii_lowercase
#   result_str = ''.join(random.choice(letters) for i in range(length))
#   return result_str

# input = get_random_string(5)

# @app.route('/')
# def hello_world():
#   return "test"


# @app.route('/image', methods=["GET"])
# def getImage():
#   storage_client = storage.Client.from_service_account_json(
#     "calhacks-365701-9935a4720153.json", project='CalHacks')
#   bucket = storage_client.get_bucket("calhacks-bucket")
#   path = os.getcwd() +'/gen-image-0.png'
#   filename = "%s%s" % ('', path)
#   blob = bucket.blob(input)
#   blob.content_type = "image/jpeg"
#   # blob.patch()

#   with open(path, 'rb') as f:
#     blob.upload_from_file(f)
#     print("Image Uploaded : ")

#   option1 = ans_choices[0]
#   option2 = ans_choices[1]
#   option3 = ans_choices[2]
#   option4 = ans_choices[3]


#   return {
#     "link":
#     "https://storage.googleapis.com/calhacks-bucket/" + input,
#     "options": [option1, option2, option3, option4],
#     "answer": answer
#   }


# app.run(host='0.0.0.0', port=8080)
