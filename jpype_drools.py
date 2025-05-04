# *_*coding:utf-8 *_*
# @Author : zyc
import jpype
import os

def startjvm():
    # 获取../jre/bin/server/jvm.dll文件位置
    # jvmPath = jpype.getDefaultJVMPath()
    jvmPath = r'jdk1.8.0_351\jre_1\bin\server\jvm.dll'
    # print(jvmPath)
    # 获取当前路径
    ext_classpath = os.getcwd()
    # 用分号连接所有jar包
    jar_path = r'./jar/drools_reasoning.jar'
    dependency = os.path.join(os.path.abspath('.'), r'jar/drools_reasoning_jar')
    # 启动jvm虚拟机，加载所有jar包，每个jar包用分号连接;解决乱码问题
    jpype.startJVM(jvmPath, "-ea", '-Djava.class.path={}\\{}'.format(ext_classpath, jar_path),
                   "-Djava.ext.dirs=%s" % dependency,"-Dfile.encoding=UTF-8")


def jpype_run_drools(filepath,save_dir):
    # 加载java类
    jClass = jpype.JClass("runall")
    jclass = jClass()  # 实例化对象
    pipe_str=jclass.excel_pipe_drools(filepath)
    print(pipe_str)
    with open(str(save_dir)+"/drools.txt","w",encoding='UTF-8') as f:
        f.write(str(pipe_str))
    f.close()



