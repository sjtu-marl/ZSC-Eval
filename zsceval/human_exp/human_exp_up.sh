IP=127.0.0.1
PORT=60088
mkdir -p data

uwsgi --http 0.0.0.0:${PORT} --module zsceval.human_exp.overcooked-flask.app:app --env POLICY_POOL=zsceval/policy_pool --env FLASK_ENV=production  --processes 1 --threads 20 --env FLASK_PORT=${PORT} --env FLASK_ACCESS_HOST=${IP} --master
# --logto data/uwsgi.log
