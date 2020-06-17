import xlwt
import xlrd


# 序号从0开始，表头第0行,有效数据从第1行开始
# 输入数据格式:1学号，2成绩，4成绩类型,3学分
std_num_index = 0
xf_index = 2 # 学分索引
score_index = 1 #考试成绩索引
type_index = 3 #考试成绩索引
input_data = xlrd.open_workbook('/Users/frank/Desktop/chengji4.xlsx')
score_sheet = input_data.sheet_by_index(0)
# 输出数据格式:0学号，1总学分,2平均学分总绩点，3总绩点
r_std_num_index = 0
r_all_xf_index = 1
r_all_score_index = 2
r_result_index = 3
# 当前数据条数+1
result_num = 1
# 总学分和总绩点
all_xf = 0;
all_score = 0;

def writeOneStudent(result_sheet,result_num):
    result_sheet.write(result_num, r_std_num_index, std_num)
    result_sheet.write(result_num, r_all_xf_index, all_xf)
    result_sheet.write(result_num, r_all_score_index, all_score)
    if all_xf!= 0:
        result_sheet.write(result_num, r_result_index, all_score/all_xf)
def writeTitle(result_sheet):
    result_sheet.write(0, r_std_num_index, '学生学号')
    result_sheet.write(0, r_all_xf_index, '学生总学分')
    result_sheet.write(0, r_all_score_index, '学生总绩点')
    result_sheet.write(0, r_result_index, '学生学分绩')


# 用于存储学分绩数据
result_book = xlwt.Workbook()
result_sheet = result_book.add_sheet('Result Sheet')
writeTitle(result_sheet)
# 流程:读取每行数据，叠加学分，总绩点，读取后判断下一行学号是否改变，如果改变，说明一个人的数据统计完毕，输出
# 缓存第一个人的学号
next_std_num = score_sheet.cell_value(1,std_num_index)
n = score_sheet.nrows
for row_num in range(1, n):
    print(row_num)
    # 取上一次读取到的本条数据的学号
    std_num = next_std_num
    type = score_sheet.cell_value(row_num, type_index)
    if type.strip() != ' none':

        # 学分
        xf = score_sheet.cell_value(row_num, xf_index)
        # 成绩(负数变正)
        score = score_sheet.cell_value(row_num, score_index)
        if xf != '' and not str(xf).isspace() and score != '' and not str(score).isspace():
            # 累加学分
            xf = abs(float(xf))
            all_xf+= xf
            # 累加绩点
            score = abs(float(score))
            #if score>=60 and score<90:
                #score=(score-50)/10
            #elif score>=90:
                #score=4
            #elif score<60:
                #score=0
            all_score+= xf*score
        # 取下一条数据学号
    if row_num<n-1:
        next_std_num = score_sheet.cell_value(row_num+1, std_num_index)
        if next_std_num != std_num:
            # 一个人的数据统计完毕
            # 写入记录
            writeOneStudent(result_sheet,result_num)
            result_num+=1
            # 清空累加学分和总绩点
            all_xf = 0;
            all_score = 0;
    else:
        # 末尾
        # 一个人的数据统计完毕
        # 写入记录
        writeOneStudent(result_sheet, result_num)
        result_num += 1
        # 清空累加学分和总绩点
        all_xf = 0
        all_score = 0
result_book.save('/Users/frank/Desktop/alxxx.csv')