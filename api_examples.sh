
curl -X POST localhost:5006/api/run_prior -H "Content-Type: application/json" --data '{"free_refs": {"water_level_0": "1.0"}}'

curl -X POST localhost:5006/api/run_prior -H "Content-Type: application/json" --data '{"free_refs": {"water_level_0": "6.0"}, "n": "1000"}'

curl -X POST localhost:5006/api/run_posterior -H "Content-Type: application/json" --data '{"free_refs": {"water_level_0": "0.0"}, "n": "1000", "observed": {"final_water_level": {"data": "8.0", "sd": "1.0"}}}'


curl localhost:5006/api/panes

curl -X POST localhost:5006/api/panes -H "Content-Type: application/json" --data '[{"loc": [0, 0, 4, 4], "type": "DiagramPane", "data": {"show_vars": true, "sparklines": false, "sparkdensities": false, "universe": [], "include_dependencies": false, "fit": true}}]'

curl -X POST localhost:5006/api/add_pane -H "Content-Type: application/json" --data '{"type": "PlotsPane", "data": {"fig_width": 10, "fig_height": 6, "columns": 2, "plot_type": "Custom", "subset": ["faucet", "drain", "final_water_level", "water_level"], "ref_subset": ["faucet", "drain", "final_water_level", "water_level"]}}'
