import numpy as np
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell
import xlrd    
import os
 

def compute_distance(_a, _b):
	"""
	Din lucrare distanta normalizata d este:

	d = || b - a || / ( ||b|| + ||a||)
	unde || . || este norma unui vector

	"""
	a = _a.astype(int)
	b = _b.astype(int)
	norm_diff = np.linalg.norm(b-a)
	norma = np.linalg.norm(a)
	normb = np.linalg.norm(b)
	return norm_diff / (norma + normb)

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

    #elements = np.zeros((rows.shape[0]))
    #for i in range(rows.shape[0]):
    #    elements[i] = rows[i]
    #sorted_el = np.array(sorted(elements , key=lambda k: [k[0], k[1]]))
    sorted_el = np.sort(elements)
    return sorted_el.astype(int)


database_path = 'E:\\Master An2\\disertatie\\proiect\\database\\hpatches-sequences-release'
directories = [ os.path.basename(f.path) for f in os.scandir(database_path) if f.is_dir() ]
excel_files = ['results_goldberg.xls', 'results_SIFT.xls', 'results_SURF.xls','results_ORB.xls', 'results_SIFTmax.xls']
col_text = ['search_folder', 'search_image', 'found_folder', 'found_image', 'distance', 'the detection is']
img = '1'

for path in excel_files:
    workbook = xlsxwriter.Workbook('search_'+ path)
    worksheet = workbook.add_worksheet()
    m = 0
    nr = 1
    for text in col_text:
        worksheet.write(0,m, text)
        m = m+1
    for i, seq in enumerate(directories) :
        el = find_value(path, seq, img)
        for element in el:
            row = find_row(path, seq, img)
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
                worksheet.write(nr,0,seq)
                worksheet.write(nr,1, img)
                worksheet.write(nr,2, sheet.cell(element, 2).value)
                worksheet.write(nr,3, sheet.cell(element, 1).value)
                worksheet.write(nr,4, d)
                truth = sheet.cell(element, 2).value == seq
                worksheet.write(nr,5, truth)
                nr = nr+1
    workbook.close()
    print(path)
    #pdb.set_trace()

