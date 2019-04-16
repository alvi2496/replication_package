proj_developer_amplitude = {str: [int, float]}

for line in open("./results/rq_2_b.csv"):
    project, developer, amplitude = line.strip().split(",")[0], line.strip().split(",")[1], line.strip().split(",")[3]

    idp = project + " " + developer
    if idp not in proj_developer_amplitude:
        proj_developer_amplitude[idp] = [amplitude, None]

proj_commits = {}
for line in open("./data/rq_2_a_c.data"):
    project = line.strip().split(" ")[0]
    commits = line.strip().split(" ")[2]

    if project not in proj_commits:
        proj_commits[project] = int(commits)
    else:
        proj_commits[project] += int(commits)


for line in open("./data/rq_2_a_c.data"):
    project, developer, commits = line.strip().split(" ")[0], line.strip().split(" ")[1], line.strip().split(" ")[2]

    idp = project+" "+developer
    if idp in proj_developer_amplitude:
        proj_developer_amplitude[idp][1] = float(commits)/proj_commits[project]


print("project,developer,commits,coverage")
for k, v in proj_developer_amplitude.items():
    if v[1] is not None and type(v[1]) == float:
        # print(','.join(k.split(' ')), v[0], v[1]*100)
        row = ','.join(k.split(' ')) + ',' + str(v[0]) + ',' + str(round(v[1]*100, 4))
        print(row)
