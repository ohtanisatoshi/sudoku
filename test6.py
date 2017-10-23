# coding:utf-8
import numpy as np

current_l = 0
current_r = 0
def forward():
    global current_l
    global current_r
    if current_r == 8:
        if current_l == 8:
            return False
        else:
            current_l += 1
            current_r = 0
    else:
        current_r += 1

def backward():
    global current_l
    global current_r
    if current_r == 0:
        if current_l == 0:
            return False
        else:
            current_l -= 1
            current_r = 8
    else:
        current_r -= 1

def set_number(a):
    if a[current_l][current_r] != 0:
        # 最初から番号が設定されているマスなので次のマスに進む
        r = forward()
        if r is False:
            # 右下までいったのでreturn
            return True
        r = set_number(a)
        backward()
        return r
    else:
        block_start_l = int(current_l / 3) * 3
        block_start_r = int(current_r / 3) * 3
        already_used_nums = np.hstack((a[current_l],
                                       a[:, current_r],
                                       a[block_start_l:block_start_l+3, block_start_r:block_start_r+3].flatten()))
        already_used_nums = np.unique(already_used_nums)
        for check_num in range(1, 10):
            # 横方向に同じ番号が設定されていないか確認
            if check_num in already_used_nums:
                continue

            # 同じ番号が設定されていなければマスに番号を設定して次のマスに進む
            a[current_l][current_r] = check_num
            r = forward()
            if r is False:
                return True
            r = set_number(a)
            backward()
            if r is True:
                return r

            # 次のマス以降で設定できる番号がなかった場合は0に戻して次の候補をテスト
            a[current_l][current_r] = 0

        return False



def solve_sudoku(q):
    a = np.copy(q)
    set_number(a)
    print(a)
    print(np.unique(a, axis=0).shape)
    print(np.unique(a, axis=1).shape)

def main():
    '''
    q = np.array([[0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,2,7,
                   4,0,0,6,0,8,0,0,0,
                   0,7,1,0,0,0,3,0,0,
                   2,3,8,5,0,6,4,1,9,
                   9,6,4,1,0,0,7,5,0,
                   3,9,5,0,2,7,8,0,0,
                   1,8,2,0,6,0,9,7,4,
                   0,4,6,8,1,9,2,0,5
                   ]]).astype(np.float32)
    q = np.array([[0,0,3,7,1,0,0,2,4,
                   0,6,0,0,0,9,0,0,0,
                   8,0,0,0,4,3,0,9,0,
                   0,3,8,9,0,2,0,0,0,
                   1,5,6,0,0,0,0,0,2,
                   0,4,2,1,0,8,0,7,5,
                   6,8,5,4,9,7,0,0,3,
                   0,1,0,6,0,5,7,4,8,
                   3,7,0,8,2,1,6,5,9,
                   ]]).astype(np.float32)
    '''
    q = np.array([[0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,2,8,0,
                   3,7,6,4,0,0,0,0,0,
                   7,0,0,0,0,1,0,0,0,
                   0,2,0,0,0,0,0,0,0,
                   4,0,0,3,0,0,0,0,6,
                   0,1,0,0,2,8,0,0,0,
                   0,0,0,0,0,5,0,0,0,
                   0,0,0,0,0,0,0,0,3,
                   ]]).astype(np.float32)
    q = q.reshape((9, 9))
    solve_sudoku(q)




if __name__ == '__main__':
    main()
