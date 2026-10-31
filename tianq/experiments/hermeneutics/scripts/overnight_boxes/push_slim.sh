K=~/.ssh/id_ed25519_vast
D=/mnt/c/Users/tqmen/AppData/Local/Temp/claude/--wsl-localhost-Ubuntu-home-user-projects-spar-model-hermeneutics/8f57f645-ac89-4088-8ef3-6fe1ca508eb8/scratchpad
cd ~/projects
tar czf /tmp/slim.tgz --exclude='__pycache__' --exclude='*/scratch' --exclude='.env' \
  spar-model-hermeneutics/tianq/dyl-probe-repro $(ls -d spar-model-hermeneutics/gigi/*/src) spar-model-hermeneutics/gigi/qwen/results/qwen3-235b
ls -la /tmp/slim.tgz | awk '{print "slim " $5/1e6 " MB"}'
push () { scp -q -i $K -P $1 /tmp/slim.tgz $D/box_driver.sh $D/verify_hub.py root@$2:/workspace/ && \
  ssh -i $K -p $1 root@$2 'cd /workspace && rm -rf spar-model-hermeneutics && tar xzf slim.tgz && cp .env spar-model-hermeneutics/tianq/dyl-probe-repro/.env && chmod +x box_driver.sh spar-model-hermeneutics/tianq/dyl-probe-repro/*.sh && echo "pushed to $(hostname)"' 2>&1 | grep -v "Welcome\|Have fun\|AI agents"; }
push 21375 69.7.162.133 & push 30350 82.98.239.123 & wait
