import numpy as np
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell
import xlrd    
import os
import cv2
from matplotlib import pyplot as plt
from definitios import compute_distance
import pdb


def find_row(excel_path, seq, img):
    column = 1
    wb = xlrd.open_workbook(excel_path)
    sheet = wb.sheet_by_index(0)
    
    for row_num in range(sheet.nrows):
        myCell1 = sheet.cell(row_num, column)
        myCell2 = sheet.cell(row_num, column+1)
        row_value = sheet.row_values(row_num)
        if myCell2.value == seq: 
            if myCell1.value == img:
                return row_num

def find_value(excel_path, seq, img):
    row = find_row(path, seq, img)
    wb = xlrd.open_workbook(excel_path)
    sheet = wb.sheet_by_index(0)
    rows = []
    columns = []
    elements = []
    for col in range(5,67):
        initCell = sheet.cell(row, col)
        initial_val = initCell.value
        
        for row_num in range(sheet.nrows):
            myCell = sheet.cell(row_num, col)
            if myCell.value == initial_val:
                if row_num != row:
                    if row_num not in rows:
                        rows= np.append(rows, row_num)
                        columns = np.append(columns, col)
                    elif row_num not in elements:
                        elements = np.append(elements, row_num)

    sorted_el = np.sort(elements)
    return sorted_el.astype(int)


database_path = 'E:\\Master An2\\disertatie\\proiect\\database\\hpatches-sequences-release'
directories = [ os.path.basename(f.path) for f in os.scandir(database_path) if f.is_dir() ]
#excel_files = ['results_goldberg.xls', 'results_SIFT.xls', 'results_SURF.xls','results_ORB.xls', 'results_SIFTmax.xls']
path = 'results_SURF.xls'
seq = 'i_books'
find_img = '1'
find_path = database_path + "\\" + seq + "\\" + find_img + ".ppm"
img = cv2.imread(find_path)
print ('imaginea initiala este ' + seq + '/' + find_img + '.ppm')
cv2.imshow('imaginea initiala',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


el = find_value(path, seq, find_img)
for element in el:
    row = find_row(path, seq, find_img)
    wb = xlrd.open_workbook(path)
    sheet = wb.sheet_by_index(0)
    initial = sheet.cell(row, 4)
    compared = sheet.cell(element, 4)
    val_init = np.fromstring(initial.value[1:-1], dtype=int, sep=',')
    val_comp = np.fromstring(compared.value[1:-1], dtype=int, sep=',')
    #pdb.set_trace()
    d = compute_distance(val_init, val_comp)
    if path == 'results_SIFTmax.xls':
        th = 0.7
    else:
        th = 0.6
    if(d < th):
        find_path = sheet.cell(element, 0).value
        image = cv2.imread(find_path)
        text = 'imaginea gasita este ' + sheet.cell(element, 2).value +"/" + sheet.cell(element, 1).value
        print (text)
        #pdb.set_trace()
        cv2.imshow(text,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#workbook.close()
pdb.set_trace()

