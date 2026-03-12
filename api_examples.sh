
curl -X POST localhost:5006/api/run_prior -H "Content-Type: application/json" --data '{"free_refs": {"water_level_0": "1.0"}}'

curl -X POST localhost:5006/api/run_prior -H "Content-Type: application/json" --data '{"free_refs": {"water_level_0": "6.0"}, "n": "1000"}'

curl -X POST localhost:5006/api/run_posterior -H "Content-Type: application/json" --data '{"free_refs": {"water_level_0": "0.0"}, "n": "1000", "observed": {"final_water_level": {"data": "8.0", "sd": "1.0"}}}'
