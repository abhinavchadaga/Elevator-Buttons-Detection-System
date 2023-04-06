from data_utils import read_split_file

datasets = read_split_file("data/panels/mixed/split.txt")
print(len(datasets[0]))
print(len(datasets[1]))
print(len(datasets[2]))

datasets = read_split_file("data/panels/ut_west_campus/split.txt")
print(len(datasets[0]))
print(len(datasets[1]))
print(len(datasets[2]))

print(sum([76
,10
,23
,192
,12
,88]))