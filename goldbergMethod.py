import numpy as np
from definitios import crop_image, compute_grid_points, compute_mean_level, compute_distance, compute_similarity
import pdb 
import cv2

path1 = 'MonaLisa_Wikipedia.jpg'
path2 = 'MonaLisa_WikiImages.jpg'
path3 = 'rotated.jpg'
path4 = 'Caravaggio_Wikipedia.jpg'
#path1 = 'starry_night1.jpg'
#path2 = 'starry_night2.jpg'
#path3 = 'starry_night4.jpg'

def generate_signature(img_path):

    #Step 0 read the image
    img=cv2.imread(img_path)

	#Step 1 convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = img.copy()
	
	#Step 2a determine cropping boundaries
    image_limits = crop_image(img, 5, 95)
	#print(image_limits)
	
	#Step 2b generate grid centers
    x_coords, y_coords = compute_grid_points(img, n = 9, window = image_limits)

	
	#Step 3a Compute dimensions of squares
    Pcalc = max([2.0, int(0.5 + min(img.shape)/20.)]) 
    P = Pcalc
    avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))
    for i, x in enumerate(x_coords):
        lower_x_lim = int(max([x - P/2, 0]))
        upper_x_lim = int(min([lower_x_lim + P, image.shape[0]]))
        for j, y in enumerate(y_coords):
            lower_y_lim = int(max([y - P/2, 0]))
            upper_y_lim = int(min([lower_y_lim + P, image.shape[1]]))

            avg_grey[i, j] = np.mean(image[lower_x_lim:upper_x_lim,
                                    lower_y_lim:upper_y_lim])

	#Step 3b Compute grey level mean of squares
    #pdb.set_trace()
    #avg_grey = compute_mean_level(img, x_coords, y_coords, P=Pcalc)
	
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

    return np.ravel(diff_mat).astype('int8')




#sig1 = generate_signature(path1)
#sig2 = generate_signature(path2)
#sig3 = generate_signature(path3)
#sig4 = generate_signature(path4)
#
#print (compute_distance(sig1, sig2))
#compute_similarity(compute_distance(sig1, sig2))
#print (compute_distance(sig1, sig3))
#compute_similarity(compute_distance(sig1, sig3))
#print (compute_distance(sig1, sig4))
#compute_similarity(compute_distance(sig1, sig4))
#pdb.set_trace()
