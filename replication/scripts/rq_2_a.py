design_discussions = {str(): (set(), set())}

project_designers_developers = {str(): (set(), set())}
for line in open('./data/rq_2_a.data'):
	tokens = line.strip().split(" ")
	activity, category, project = tokens[1], tokens[2], tokens[3]

	id_activity = project+" "+activity

	if id_activity not in design_discussions:
		if category == "design":
			design_discussions[id_activity] = (set(), set())
			project_designers_developers[project] = (set(), set())


# designers to first slot
for line in open('./data/rq_2_a.data'):
	tokens = line.strip().split(" ")
	developer, activity, project = tokens[0], tokens[1], tokens[3]

	id_activity = project+" "+activity

	if id_activity in design_discussions:
		project_designers_developers[project][0].add(developer)

for line in open('./data/rq_2_a_c.data'):
	tokens = line.strip().split(" ")
	project, developer = tokens[0], tokens[1]
	if project in project_designers_developers:
		project_designers_developers[project][1].add(developer)

print("project_name,designer_contributors,all_contributors,proportion")
for k, v in project_designers_developers.items():
	if len(v[1]) + len(v[0].intersection(v[1])) > 0:
		row = str(k) + ',' + str(len(v[0].intersection(v[1]))) + ',' + str(len(v[1])) + ',' + \
				str(float(len(v[0].intersection(v[1]))) / (len(v[0].intersection(v[1])) + len(v[1])))
		print(row)
