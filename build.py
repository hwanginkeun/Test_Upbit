import PyInstaller.__main__
import os
import shutil

def build_exe():
    # 빌드 디렉토리 생성
    if not os.path.exists('build'):
        os.makedirs('build')
    if not os.path.exists('dist'):
        os.makedirs('dist')

    # PyInstaller 실행
    PyInstaller.__main__.run([
        'app.py',
        '--name=업비트검색기',
        '--onefile',
        '--windowed',
        '--add-data=templates;templates',
        '--icon=templates/favicon.ico',
        '--hidden-import=sklearn.utils._cython_blas',
        '--hidden-import=sklearn.neighbors.typedefs',
        '--hidden-import=sklearn.neighbors.quad_tree',
        '--hidden-import=sklearn.tree',
        '--hidden-import=sklearn.tree._utils',
    ])

    # 필요한 파일 복사
    if os.path.exists('dist'):
        # templates 폴더 복사
        templates_dist = os.path.join('dist', 'templates')
        if not os.path.exists(templates_dist):
            shutil.copytree('templates', templates_dist)

if __name__ == '__main__':
    build_exe() 