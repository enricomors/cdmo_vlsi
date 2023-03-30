import os

if __name__ == "__main__":

    dirname = f'heights'

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    out = ""
    out += '\n'.join([f"{i} & N/A & N/A & N/A & N/A \\\\" for i in range(5, 41)])
    print(out)

    with open(f'{dirname}/heights.txt', 'w') as f:
        f.write(out)
