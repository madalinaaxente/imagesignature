import cv2 
import numpy as np
from definitios import crop_image
import pdb

path1 = 'E:\\Master An2\\disertatie\\proiect\\database\\hpatches-sequences-release\\i_castle\\1.ppm'
#path2 = 'MonaLisa_WikiImages.jpg'
#path3 = 'rotated.jpg'
#path4 = 'Caravaggio_Wikipedia.jpg'
#path1 = 'starry_night1.jpg'
#path2 = 'starry_night2.jpg'
#path3 = 'starry_night4.jpg'

def interval_divide(min, max, intervals):
   assert intervals != 0
   interval_size = round((max - min) / intervals)
   result = []
   start = min
   end = min + interval_size
   while True:
      result.append([start, end])
      start = end + 1
      end = end + interval_size
      if len(result) == intervals:
         break
   return result

def generate_SURFsignature(img_path):

    # reading the image
    img = cv2.imread(img_path)
    
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    window = crop_image(gray, 5, 95)

    if window is None:
        img = img[(0, img.shape[0]), (0, img.shape[1]),:]
    else:
        img = img[window[0][0]:window[0][1],window[1][0]:window[1][1],:]

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    image2 = img.copy()
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create SURF feature extractor
    surf = cv2.xfeatures2d.SURF_create()

    # detect features from the image
    keypoints, descriptors = surf.detectAndCompute(image2, None)

    pts = np.asarray([[p.pt[0], p.pt[1]] for p in keypoints])
    response = np.asarray(list(p.response for p in keypoints))

    interval_no = 9
    #divide the image in 9 equal intervals
    length_intervals = interval_divide(0, img.shape[0], interval_no)
    width_intervals = interval_divide(0, img.shape[1], interval_no)

    x_coords = []
    y_coords = []
    pts_idx = np.array([])

    for width in width_intervals:
        for length in length_intervals:
            
            #points = pts[(pts[:,0]>width[0]) & (pts[:,0]< width[1])]
            points_idx = np.array(np.where((pts[:,0]>width[0]) & (pts[:,0]< width[1])))
            #points_y =  points[(points[:,1]>length[0]) & (points[:,1] < length[1])]
            points_idx_y = np.array(np.where((pts[:,1]>length[0]) & (pts[:,1] < length[1])))
            if points_idx.size and points_idx_y.size:
                pts_idx = np.intersect1d(points_idx, points_idx_y)

            if pts_idx.size:
                resp = response[pts_idx[:]]
                res = np.array(np.where(resp == np.max(resp)))
                x_coord,y_coord = pts[pts_idx[res[0][0]]]
                if(np.max(resp)!=response[pts_idx[res[0][0]]]):
                    print("eroare!!!!!!!!!")
                x_coords = np.append(x_coords, round(x_coord))
                y_coords = np.append(y_coords, round(y_coord))

            else:
                x_coord = np.linspace(width[0], width[1], 3, dtype=int)[1:-1]
                y_coord = np.linspace(length[0], length[1], 3, dtype=int)[1:-1]
                x_coords = np.append(x_coords, x_coord)
                y_coords = np.append(y_coords, y_coord)

    x_coords= x_coords.astype(int)
    y_coords= y_coords.astype(int)
    #Step 3a Compute dimensions of squares
    P = max([2.0, int(0.5 + min(gray.shape)/20.)]) 

    nr = 0
    avg_grey = np.zeros((y_coords.shape[0]))
    for i, y in enumerate(y_coords):
        lower_y_lim = int(max([y - P/2, 0]))
        upper_y_lim = int(min([lower_y_lim + P, img.shape[0]]))
        lower_x_lim = int(max([x_coords[i] - P/2, 0]))
        upper_x_lim = int(min([lower_x_lim + P, img.shape[1]]))
        avg_grey[i] = np.mean(gray[lower_y_lim:upper_y_lim,lower_x_lim:upper_x_lim])
        nr = nr+1
        end_point = (upper_x_lim, upper_y_lim)
        start_point = (lower_x_lim,lower_y_lim)
        cv2.rectangle(image,start_point,end_point,(0,255,0),2)

    cv2.imwrite("nr2/image_1castlesurf.jpg", image) 

    avg_grey = avg_grey.reshape(interval_no, interval_no)

    #Step 4a Compute array of differences for each grid point
    grey_level_matrix = avg_grey

    right_neighbors = -np.concatenate((np.diff(grey_level_matrix),
									np.zeros(grey_level_matrix.shape[0]).
									reshape((grey_level_matrix.shape[0], 1))),
									axis=1)
	# pentru a calcula valorile vecinilor din dreapta facem diferenta cu valoarea din stanga 
	# (deci punem minus in fata) pentru toate nr, iar pentru ultima coloana punem padding cu 0 
	# 	
    left_neighbors = -np.concatenate((right_neighbors[:, -1:],
									right_neighbors[:, :-1]),
									axis=1)
	# pentru vecinii din stanga
	# diferenta este opusul celui din dreapta (de asta punem - in fata) si vom avea 0 pe primul rand, nu pe ultimul

    down_neighbors = -np.concatenate((np.diff(grey_level_matrix, axis=0),
									np.zeros(grey_level_matrix.shape[1]).
									reshape((1, grey_level_matrix.shape[1]))))
	# pentru vecinii de jos calculam
	# diferenta dintre vecinul de deasupra cu minus -> diferenta cu cel de dedesupt, padding cu 0 pe ultima linie

    up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))
	#pentru vecinii de sus
	#diferenta este opusul celui de jos (cu minus) cu padding pe prima linie, nu pe ultima

    diagonals = np.arange(-grey_level_matrix.shape[0] + 1, grey_level_matrix.shape[0])
	#returneaza numere din intervalul [-nr_col, nr_col] in ordine

    upper_left_neighbors = sum([np.diagflat(np.insert(
					np.diff(np.diag(grey_level_matrix, i)), 0, 0), i)
					for i in diagonals])
	#np.diag(maxtrix, i) -> diagonala principala este 0, iar celelalte sunt cu minus in stanga jos si cu plus in dr sus pana la nr_col
	#np.diff face diferentele dintre elementele pe diagonala
	#np.insert insereaza 0 in fata elemtului 0 (al doilea parameteru este indicele elem = 0, al treilea este valoarea = 0)
	# np.diagflat creeaza o matrice punanand e diagonala i valorile primite
	# toate matricele se aduna apoi pt a creea matricea completa

    lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:], (0, 1), mode='constant')
	#matricea upper left cu minus si padding in ultimul rand si ultima coloana, asa ca se vor ignora primul rand si prima coloana 

    flipped = np.fliplr(grey_level_matrix)
	#matricea in oglinda pt a aplica exact acelasi algoritm ca mai sus, dar oglindit
    upper_right_neighbors = sum([np.diagflat(np.insert(
					np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
	#le oglindim apoi pt a avea valorile corecte
    upper_right_neighbors = np.fliplr(upper_right_neighbors)

    lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:],
											(0, 1), mode='constant')
    lower_left_neighbors = np.fliplr(lower_left_neighbors)
	
    diff_mat = np.dstack(np.array([upper_left_neighbors, 
					up_neighbors,
					upper_right_neighbors,
					left_neighbors,
					right_neighbors,
					lower_left_neighbors,
					down_neighbors,
					lower_right_neighbors]))
	
	
	#Step 4b normalize the matrices of difference
    identical_tolerance=2/255. #diferenta maxima pentru ca doua valori de gri sa fie considerate la fel
    n_levels=2 #nr de bins = 2n+1 (n = 2 => [-2,-1,0,1,2])
	
    mask = np.abs(diff_mat) < identical_tolerance
    diff_mat[mask] = 0.
	
    if np.all(mask):
    	pass
    else:
    	positive_cutoffs = np.percentile(diff_mat[diff_mat > 0.],
							np.linspace(0, 100, n_levels+1))
		#se aleg doar valorile poz din diff_mat
		#np.linspace(0, 100, n_levels+1) -> n_levels+1 de la 0 la 100
		#vom avea n_levels+1 numere

    	negative_cutoffs = np.percentile(diff_mat[diff_mat < 0.],
							np.linspace(100, 0, n_levels+1))
    	for level, interval in enumerate([positive_cutoffs[i:i+2] for i in range(positive_cutoffs.shape[0] - 1)]):

            diff_mat[(diff_mat >= interval[0]) & (diff_mat <= interval[1])] = level + 1
		
    	for level, interval in enumerate([negative_cutoffs[i:i+2] for i in range(negative_cutoffs.shape[0] - 1)]):

            diff_mat[(diff_mat <= interval[0]) & (diff_mat >= interval[1])] = -(level + 1)
            
    #pdb.set_trace()
    return np.ravel(diff_mat).astype('int8')

sig1 = generate_SURFsignature(path1)
#sig2 = generate_SURFsignature(path2)
#sig3 = generate_SURFsignature(path3)
#sig4 = generate_SURFsignature(path4)
#
#print (compute_distance(sig1, sig2))
#compute_similarity(compute_distance(sig1, sig2))
#print (compute_distance(sig1, sig3))
#compute_similarity(compute_distance(sig1, sig3))
#print (compute_distance(sig1, sig4))
#compute_similarity(compute_distance(sig1, sig4))
##pdb.set_trace()

