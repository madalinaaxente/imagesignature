import numpy as np
import os
from goldbergMethod import generate_signature
from goldbergSIFTmax import generate_SIFTMAXsignature
from goldbergSIFT import generate_SIFTsignature
from goldbergSURF import generate_SURFsignature
from goldbergORB import generate_ORBsignature
from definitios import compute_distance
import pdb
import xlsxwriter
from pathlib import Path


signature_type = ['goldberg', 'SURF', 'SIFT', 'ORB', 'SIFTmax']
#signature_type = ['SIFTmax']

def words_to_int(word_array):
    """Converteste cuvinterle in numere intregi
    Considera intrarea un numar in baza 3 si il convereste in baza 10
    Adauga 1 penru a trasnforma valorile in baza 3 {0,1,2}

    """
    width = word_array.shape[1]

    # Three states (-1, 0, 1)
    coding_vector = 3**np.arange(width)

    # The 'plus one' here makes all digits positive, so that the
    # integer representation is strictly non-negative and unique
    return np.dot(word_array + 1, coding_vector)

def create_record(img_path, signature_type, k, N, ):
    """Creează un item ce va fi inserat in excel 
    impartind semnatura in mai multe cuvinte
    """
    record = dict()
    metadata = os.path.basename(os.path.dirname(img_path))
    record['path'] = img_path
    record['manipulation_type'] = os.path.splitext(os.path.basename(img_path))[0]
    record['metadata'] = metadata
    record['signature_type'] = signature_type
    if (signature_type == 'goldberg'):
        signature = generate_signature(img_path)
    elif(signature_type == 'SIFTmax'):
        signature = generate_SIFTMAXsignature(img_path)
    elif(signature_type == 'SURF'):
        signature = generate_SURFsignature(img_path)
    elif(signature_type == 'SIFT'):
        signature = generate_SIFTsignature(img_path)
    elif(signature_type == 'ORB'):
        signature = generate_ORBsignature(img_path)
    else:
        raise ValueError('Signature type is not valid')

    record['signature'] = signature.tolist()

    word_positions = np.linspace(0, signature.shape[0], N, endpoint=False).astype('int')
    
    if k > signature.shape[0]:
        raise ValueError('Word length cannot be longer than array length')
    if word_positions.shape[0] > signature.shape[0]:
        raise ValueError('Number of words cannot be more than array length')

    words = np.zeros((N, k)).astype('int8')

    for i, pos in enumerate(word_positions):
        if pos + k <= signature.shape[0]:
            words[i] = signature[pos:pos+k]
        else:
            temp = signature[pos:].copy()
            temp.resize(k)
            words[i] = temp

    #normalzie words to interval [-1,1]
    words[words > 0] = 1
    words[words < 0] = -1

    words = words_to_int(words)

    for i in range(N):
        record[''.join(['simple_word_', str(i)])] = words[i].tolist()

    return record

def create_distance_record(ref_img_path, img_path, signature_type):
    """
    Creează un item ce va fi inserat in excel cu urmatoarele coloane:
    'img_path','ref_img_path' 'manipulation_type', 'metadata', 'signature', 'distance', 'similarity'

    """
    record = dict()
    record['img_path'] = img_path
    record['ref_img_path'] = ref_img_path
    record['manipulation_type'] = os.path.splitext(os.path.basename(img_path))[0]
    record['metadata'] = os.path.basename(os.path.dirname(img_path))
    if (signature_type == 'goldberg'):
        signature = generate_signature(img_path)
        ref_signature = generate_signature(ref_img_path)
    elif(signature_type == 'SIFTmax'):
        signature = generate_SIFTMAXsignature(img_path)
        ref_signature = generate_SIFTMAXsignature(ref_img_path)
    elif(signature_type == 'SURF'):
        signature = generate_SURFsignature(img_path)
        ref_signature = generate_SURFsignature(ref_img_path)
    elif(signature_type == 'SIFT'):
        signature = generate_SIFTsignature(img_path)
        ref_signature = generate_SIFTsignature(ref_img_path)
    elif(signature_type == 'ORB'):
        signature = generate_ORBsignature(img_path)
        ref_signature = generate_ORBsignature(ref_img_path)
    else:
        raise ValueError('Signature type is not valid')
    record['distance'] = compute_distance(ref_signature, signature)
    record['similarity'] =  record['distance'] < 0.6
    return record

    
def create_false_distance_record(ref_img_path, img_path, signature_type):
    """
    Creează un item ce va fi inserat in excel cu urmatoarele coloane:
    'ref_img_path','img_path' , 'metadata_ref', 'metadata 2nd image', 'distance from ref', 'similarity'

    """
    record = dict()
    record['ref_img_path'] = ref_img_path
    record['img_path'] = img_path
    record['manipulation_type'] = os.path.splitext(os.path.basename(img_path))[0]
    record['metadata_ref'] = os.path.basename(os.path.dirname(ref_img_path))
    record['metadata_img'] = os.path.basename(os.path.dirname(img_path))
    if (signature_type == 'goldberg'):
        signature = generate_signature(img_path)
        ref_signature = generate_signature(ref_img_path)
    elif(signature_type == 'SIFTmax'):
        signature = generate_SIFTMAXsignature(img_path)
        ref_signature = generate_SIFTMAXsignature(ref_img_path)
    elif(signature_type == 'SURF'):
        signature = generate_SURFsignature(img_path)
        ref_signature = generate_SURFsignature(ref_img_path)
    elif(signature_type == 'SIFT'):
        signature = generate_SIFTsignature(img_path)
        ref_signature = generate_SIFTsignature(ref_img_path)
    elif(signature_type == 'ORB'):
        signature = generate_ORBsignature(img_path)
        ref_signature = generate_ORBsignature(ref_img_path)
    else:
        raise ValueError('Signature type is not valid')
    record['distance'] = compute_distance(ref_signature, signature)
    record['similarity'] =  record['distance'] < 0.6
    return record



def insert_image_record(rec, nr, worksheet):
    """Insereaza intrun excel intrarea rec pe randul nr, 
    pe cate o coloana separata

    """
    #workbook = xlsxwriter.Workbook('results.xlsx' )
    #worksheet = workbook.add_worksheet()
    i = 0
    for key, value in rec.items():
        #worksheet.write(0,i, key)
        if( isinstance(value, list) ):
            worksheet.write(nr+1,i,str(value))
        else:
            worksheet.write(nr+1,i,value)
        i = i + 1


def add_image(img_path, signature_type, nr, worksheet):
    """Adauga o imagine

    """
    k = 16
    N = 63
    rec = create_record(img_path, signature_type, k, N)
    insert_image_record(rec, nr, worksheet)

def add_image_distance(ref_img_path, img_path, signature_type, nr):

    rec = create_distance_record(ref_img_path, img_path, signature_type)
    insert_image_record(rec, nr, worksheet)

def add_false_image_distance(ref_img_path, img_path, signature_type, nr):

    rec = create_false_distance_record(ref_img_path, img_path, signature_type)
    insert_image_record(rec, nr, worksheet)


def search_image(img_path):
    """Search image matches
    
    """



img_path1 = 'E:\\Master An2\\disertatie\\proiect\\disertatie-cod\\MonaLisa_Wikipedia.jpg'
rec1 = create_record(img_path1, 'SIFT', 16, 63)
database_path = 'E:\\Master An2\\disertatie\\proiect\\database\\hpatches-sequences-release'
directories = [ os.path.basename(f.path) for f in os.scandir(database_path) if f.is_dir() ]
gen_signature = 0
gen_distance = 0
gen_false_distance = 1

if(gen_signature):
    for i, signature_t in enumerate(signature_type):
        workbook = xlsxwriter.Workbook('2results_'+ signature_t+'.xlsx')
        worksheet = workbook.add_worksheet()
        m = 0
        for key, value in rec1.items():
            worksheet.write(0,m, key)
            m = m+1
        l = len(rec1)
        for j, folder in enumerate(directories) :
            path = database_path + "\\"+ folder
            for r, filename in enumerate(Path(path).glob('*.ppm')):
                nr = j*20+r
                add_image(str(filename), signature_t, nr, worksheet)

        workbook.close()
        #pdb.set_trace()


col_text = ['img_path','ref_img_path' 'manipulation_type', 'metadata', 'signature', 'distance from ref', 'similarity']
if(gen_distance):
    for i, signature_t in enumerate(signature_type):
        workbook = xlsxwriter.Workbook('2dist_results_'+ signature_t+'.xlsx')
        worksheet = workbook.add_worksheet()
        m = 0
        for text in col_text:
            worksheet.write(0,m, text)
            m = m+1
        l = len(rec1)
        for j, folder in enumerate(directories) :
            path = database_path + "\\"+ folder
            ref_file = path + "\\1.ppm"
            print(path)
            for r, filename in enumerate(Path(path).glob('*.ppm')):
                nr =j*20+r
                add_image_distance(str(ref_file), str(filename), signature_t, nr)

        workbook.close()
        #pdb.set_trace()


col_text2 = ['ref_img_path','img_path' , 'metadata_ref', 'metadata 2nd image', 'distance from ref', 'similarity']
if(gen_false_distance):
    for i, signature_t in enumerate(signature_type):
        workbook = xlsxwriter.Workbook('fin_dist_false_results_'+ signature_t+'.xlsx')
        worksheet = workbook.add_worksheet()
        m = 0
        for text in col_text2:
            worksheet.write(0,m, text)
            m = m+1
        l = len(rec1)
        nr = 0
        for j1, folder1 in enumerate(directories) :
            ref_file = database_path + "\\"+ folder1 + "\\1.ppm"
            print(database_path + "\\"+ folder1)
            for j2, folder2 in enumerate(directories[j1+1:]) :
                filename = database_path + "\\"+ folder2 + "\\1.ppm"
                add_false_image_distance(str(ref_file), str(filename), signature_t, nr)
                nr = nr +1

        workbook.close()
        #pdb.set_trace()



pdb.set_trace()
