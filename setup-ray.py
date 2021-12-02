"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
import setuptools
import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
README = os.path.join(CUR_DIR, "README.md")
with open("README.md", "r") as fd:
    long_description = fd.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()

# 打包的目录
packages = ["ai_code"] + ["%s.%s" % ("ai_code", i) for i in setuptools.find_packages("ai_code")]

setuptools.setup(
    name="ai-code",
    # 版本，不用修改，在aci流程中可以自动修改
    version="0.0.1",
    description="AI CODE",
    long_description="AI CODE",
    long_description_content_type="text/markdown",
    url="http://finmodelops.alipay.com",
    # TODO: 修改用户
    author="admin",
    # TODO: 修改用户邮箱
    author_email="admin@antgroup.com",
    packages=packages,
    include_package_data=True,
    install_requires=required,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 1 - Release',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: ANT_FIN :: ANT_FIN_GROUP',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Programming Language :: Python",
    ],
    keywords='demo',
    project_urls={  # 可选
        # TODO: 可以写自己git仓库地址
        'Bug Reports': 'https://code.alipay.com/finalgo/ai-code',
        'Source': 'https://code.alipay.com/finalgo/ai-code',
    },
)
