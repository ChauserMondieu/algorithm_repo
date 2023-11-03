class NQueen:
    # store column position
    # class variable
    row_list = []
    cnt = 0

    def __init__(self, queens=8):
        # number of queens in this instance
        # instance variable
        self.queens = queens
        NQueen.row_list = [0] * (self.queens + 1)

    def is_place(self, row, col):
        for i in range(1, row + 1):
            if NQueen.row_list[i] == col:
                return False
            if i + NQueen.row_list[i] == row + col:
                return False
            if i - NQueen.row_list[i] == row - col:
                return False
        return True

    def show_res(self):
        print("This result is: ")
        for row, col in enumerate(NQueen.row_list):
            print("%s, %s" % (row, col), end='\t')
        print()

    def dfs(self, row):
        if row == self.queens + 1:
            NQueen.cnt += 1
            self.show_res()
        for i in range(1, self.queens + 1):
            if self.is_place(row, i):
                NQueen.row_list[row] = i
                self.dfs(row + 1)
                NQueen.row_list[row] = 0

    def main(self):
        print("start algorithm ...")
        self.dfs(1)
        print("===================")
        print("total possible results are: ", NQueen.cnt)


if __name__ == "__main__":
    nqueen = NQueen(8)
    nqueen.main()
