import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
])

# df = pd.read_csv('../data/val.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Category', 'Claim OCR', 'Document OCR']]
df = pd.read_csv('../data/test.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Claim OCR', 'Document OCR']]

print(df.shape)

category = {
    'Support_Multimodal': 0,
    'Support_Text': 1,
    'Insufficient_Multimodal': 2,
    'Insufficient_Text': 3,
    'Refute': 4
}

# df['Label'] = df['Category'].map(category)

data, ids = {}, []
for n, row in tqdm(df.iterrows(), total=df.shape[0]):
    # path = '../data/image/images-train/'
    # path = '../data/image/images-val/'
    path = '../data/image/images-test/'
    # filename = path + 'claim/' + row['Category'] + '/' + str(n) + '.jpg'
    filename = path + 'claim/' + '/' + str(n) + '.jpg'
    input_claim_image = Image.open(filename)
    claim_image = preprocess(input_claim_image)

    # filename = path + 'document/' + row['Category'] + '/' + str(n) + '.jpg'
    filename = path + 'document/' + '/' + str(n) + '.jpg'
    input_document_image = Image.open(filename)
    document_image = preprocess(input_document_image)

    # data[n] = (row['claim'], claim_image, row['document'], document_image, row['Label'], row['Claim OCR'], row['Document OCR'])
    data[n] = (row['claim'], claim_image, row['document'], document_image, row['Claim OCR'], row['Document OCR'])

with open('processed_test.pickle', 'wb') as file:
    pickle.dump(data, file)