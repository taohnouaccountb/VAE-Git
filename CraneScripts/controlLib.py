import os
import time
import re
################## ENVIRONMENT VAR ###################
username = 'tyao'
project_path = '/work/cse496dl/'+username+'/HW3/'
gpu_switch = True
#######################################################
# Prepare
ltime = time.localtime()
prefix = str(ltime.tm_mon)+'_'+str(ltime.tm_mday)+'_'
prefix += str(ltime.tm_hour)+'_'+str(ltime.tm_min)+'_'+str(ltime.tm_sec)
count =0
jobid = -1
print('PREFIX: '+prefix)

# Load functions
def check_status(id):
    os.system('squeue -u '+username+' |grep '+jobid+' >'+project_path+'scripts/tempQueue.txt')
    with open(project_path+'scripts/tempQueue.txt', 'r') as file:
        rst_q=file.read()
    return rst_q

def auto_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def launch_job(gpu=True):
    global project_path, prefix, count, jobid, output_dir, script_dir, output_path, error_path, models, gpu_switch
    gpu_switch=gpu
    output_dir=project_path+'output/'+prefix+'-'+str(count)+'/'
    script_dir=project_path+'scripts/'+prefix+'-'+str(count)+'/' # ./scripts/hh_mm_ss-count/
    output_path = output_dir+prefix+'.out'
    error_path = output_dir+prefix+'.err'
    
    count = count+1
    
    # Create directories and backup py files of current version
    auto_mkdir(project_path+'output/')
    auto_mkdir(output_dir)
    auto_mkdir(project_path+'scripts/')
    auto_mkdir(script_dir)
    os.system('cp *.py '+script_dir)

    # Prepare .sh
    commands = []
    commands.append('#!/bin/sh')
    commands.append('#SBATCH --time=01:00:00')
    commands.append('#SBATCH --mem=32000')
    commands.append('#SBATCH --job-name=DDF2')
    commands.append('#SBATCH --partition=gpu')
    commands.append('#SBATCH --gres=gpu')
    commands.append('#SBATCH --constraint=[gpu_p100|gpu_k40|gpu_k20]')
    commands.append('#SBATCH --error='+error_path)
    commands.append('#SBATCH --output='+output_path)
    commands.append('')    
    if gpu_switch:
        commands.append('module load singularity')
        commands.append('singularity exec docker://unlhcc/tensorflow-gpu python3 -u $@')
    else:
        commands.append('module load tensorflow/py36/1.5')
        commands.append('echo Load Finished')
        commands.append('python3 -u $@')
    
    # Wrtie .sh
    with open(script_dir+'run.sh', 'w') as file:
        for cmd in commands:
            file.write(cmd+'\n')

    # Launch Job
    sig = os.system('sbatch '+script_dir+'run.sh '+script_dir+'/main.py >> '+script_dir+'launch_command_result.txt')
    if sig==0:
        print('Success')
        with open(script_dir+'launch_command_result.txt', 'r') as file:
            jobid = re.search('[0-9]+',file.read()).group(0)
        print('JobID '+jobid)
    else:
        print('Faild')

    # Gathering Model File
    with open(output_dir+'.model','w') as file:
        with open(script_dir+'model.py', 'r') as file2:
            model_py=file2.read()
        models = re.search('def encoder_bundle[\s\S]+return',model_py)
        file.write(models.group(0)+'\n')

def print_error():
    global output_path
    with open(error_path,'r') as f:
        s=f.read()
        print(s)

def print_output():
    global output_path
    with open(output_path,'r') as f:
        interval = 3
        while(True):
            s=f.read()
            if not s=='':
                print(s)
            if interval==0:
                if not still_running():
                    print("===STOP RUNNING===")
                    print_error()
                    break
                interval =3
            else:
                interval=interval-1
            time.sleep(5)



def still_running():
    global jobid
    rst_q = check_status(jobid)
    if ((not rst_q=='') or (rst_q.find('CG')!=-1)):
        return True
    else:
        return False
            
def print_status():
    global jobid
    if still_running():
        print(check_status(jobid))
    else:
        print('Job Finished/Terminated')
    
def cancel_job():
    global jobid
    sig = os.system('scancel '+jobid)
    print_status()

    
def print_models():
    print(models.group(0))
