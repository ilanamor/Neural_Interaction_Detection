import os
import pathlib
import shutil
import subprocess
import sys

import click
import dotenv

import pyqt5_tools.badplugin
import pyqt5_tools.examplebuttonplugin


here = pathlib.Path(__file__).parent
example_path = str(
    pathlib.Path(pyqt5_tools.examplebuttonplugin.__file__).parent,
)
bad_path = str(
    pathlib.Path(pyqt5_tools.badplugin.__file__).parent,
)

def pyqt5toolsinstalluic():
    destination = here/'bin'
    destination.mkdir(parents=True, exist_ok=True)
    there = pathlib.Path(sys.executable).parent

    shutil.copy(str(there/'pyuic5.exe'), str(destination/'uic.exe'))


def load_dotenv():
    env_path = dotenv.find_dotenv()
    if len(env_path) > 0:
        os.environ['DOT_ENV_DIRECTORY'] = str(pathlib.Path(env_path).parent)
        dotenv.load_dotenv(dotenv_path=env_path, override=True)


@click.command(
    context_settings={
        'ignore_unknown_options': True,
        'allow_extra_args': True,
    },
)
@click.pass_context
@click.option(
    '--widget-path',
    '-p',
    'widget_paths',
    help='Paths to be combined with PYQTDESIGNERPATH',
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    multiple=True,
)
@click.option(
    '--example-widget-path',
    help='Include the path for the pyqt5-tools example button ({})'.format(
        example_path,
    ),
    is_flag=True,
)
@click.option(
    '--designer-help',
    help='Pass through to get Designer\'s --help',
    is_flag=True,
)
@click.option(
    '--test-exception-dialog',
    help='Raise an exception to check the exception dialog functionality.',
    is_flag=True,
)
def pyqt5designer(
        ctx,
        widget_paths,
        designer_help,
        example_widget_path,
        test_exception_dialog,
):
    load_dotenv()

    extras = []
    widget_paths = list(widget_paths)

    if designer_help:
        extras.append('--help')

    if example_widget_path:
        widget_paths.append(example_path)

    if test_exception_dialog:
        widget_paths.append(bad_path)

    env = dict(os.environ)

    env['PYQTDESIGNERPATH'] = (
        os.pathsep.join((
            *widget_paths,
            env.get('PYQTDESIGNERPATH', ''),
            '',
        ))
    )
    env['PYTHONPATH'] = (
        os.pathsep.join((
            *sys.path,
            env.get('PYTHONPATH', ''),
            '',
        ))
    )
    env['PATH'] = (
        os.pathsep.join((
            str(pathlib.Path(sys.executable).parent),
            env.get('PATH', ''),
            '',
        ))
    )

    for name in ('PYQTDESIGNERPATH', 'PYTHONPATH', 'PATH'):
        print('{}: {}'.format(name, env[name]))

    command = [
        str(here / 'designer.exe'),
        *extras,
        *ctx.args,
    ]

    return subprocess.call(command, env=env)


qml2_import_path_option = click.option(
    '--qml2-import-path',
    '-p',
    'qml2_import_paths',
    help='Paths to be combined with QML2_IMPORT_PATH',
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    multiple=True,
)


@click.command(
    context_settings={
        'ignore_unknown_options': True,
        'allow_extra_args': True,
    },
)
@click.pass_context
@qml2_import_path_option
@click.option(
    '--qmlscene-help',
    help='Pass through to get QML scene\'s --help',
    is_flag=True,
)
def pyqt5qmlscene(
        ctx,
        qml2_import_paths,
        qmlscene_help,
):
    load_dotenv()
    extras = []

    if qmlscene_help:
        extras.append('--help')

    env = dict(os.environ)

    env['QML2_IMPORT_PATH'] = (
        os.pathsep.join((
            *qml2_import_paths,
            env.get('QML2_IMPORT_PATH', ''),
            '',
        ))
    )
    env['PYTHONPATH'] = (
        os.pathsep.join((
            *sys.path,
            env.get('PYTHONPATH', ''),
            '',
        ))
    )

    for name in ('QML2_IMPORT_PATH', 'PYTHONPATH'):
        print('{}: {}'.format(name, env[name]))

    command = [
        str(here / 'qmlscene.exe'),
        *extras,
        *ctx.args,
    ]

    return subprocess.call(command, env=env)


@click.command(
    context_settings={
        'ignore_unknown_options': True,
        'allow_extra_args': True,
    },
)
@click.pass_context
@qml2_import_path_option
@click.option(
    '--qmltestrunner-help',
    help='Pass through to get QML test runner\'s --help',
    is_flag=True,
)
def pyqt5qmltestrunner(
        ctx,
        qml2_import_paths,
        qmltestrunner_help,
):
    load_dotenv()
    extras = []

    if qmltestrunner_help:
        extras.append('--help')

    env = dict(os.environ)

    env['QML2_IMPORT_PATH'] = (
        os.pathsep.join((
            *qml2_import_paths,
            env.get('QML2_IMPORT_PATH', ''),
            '',
        ))
    )
    env['PYTHONPATH'] = (
        os.pathsep.join((
            *sys.path,
            env.get('PYTHONPATH', ''),
            '',
        ))
    )

    for name in ('QML2_IMPORT_PATH', 'PYTHONPATH'):
        print('{}: {}'.format(name, env[name]))

    command = [
        str(here / 'qmltestrunner.exe'),
        *extras,
        *ctx.args,
    ]

    return subprocess.call(command, env=env)


# def designer():
#     return subprocess.call([str(here/'designer.exe'), *sys.argv[1:]])

def assistant():
    load_dotenv()
    return subprocess.call([str(here/'assistant.exe'), *sys.argv[1:]])


def canbusutil():
    load_dotenv()
    return subprocess.call([str(here/'canbusutil.exe'), *sys.argv[1:]])


def designer():
    load_dotenv()
    return subprocess.call([str(here/'designer.exe'), *sys.argv[1:]])


def dumpcpp():
    load_dotenv()
    return subprocess.call([str(here/'dumpcpp.exe'), *sys.argv[1:]])


def dumpdoc():
    load_dotenv()
    return subprocess.call([str(here/'dumpdoc.exe'), *sys.argv[1:]])


def lconvert():
    load_dotenv()
    return subprocess.call([str(here/'lconvert.exe'), *sys.argv[1:]])


def linguist():
    load_dotenv()
    return subprocess.call([str(here/'linguist.exe'), *sys.argv[1:]])


def lrelease():
    load_dotenv()
    return subprocess.call([str(here/'lrelease.exe'), *sys.argv[1:]])


def lupdate():
    load_dotenv()
    return subprocess.call([str(here/'lupdate.exe'), *sys.argv[1:]])


def pixeltool():
    load_dotenv()
    return subprocess.call([str(here/'pixeltool.exe'), *sys.argv[1:]])


def qcollectiongenerator():
    load_dotenv()
    return subprocess.call([str(here/'qcollectiongenerator.exe'), *sys.argv[1:]])


def qdbus():
    load_dotenv()
    return subprocess.call([str(here/'qdbus.exe'), *sys.argv[1:]])


def qdbuscpp2xml():
    load_dotenv()
    return subprocess.call([str(here/'qdbuscpp2xml.exe'), *sys.argv[1:]])


def qdbusviewer():
    load_dotenv()
    return subprocess.call([str(here/'qdbusviewer.exe'), *sys.argv[1:]])


def qdbusxml2cpp():
    load_dotenv()
    return subprocess.call([str(here/'qdbusxml2cpp.exe'), *sys.argv[1:]])


def qdoc():
    load_dotenv()
    return subprocess.call([str(here/'qdoc.exe'), *sys.argv[1:]])


def qgltf():
    load_dotenv()
    return subprocess.call([str(here/'qgltf.exe'), *sys.argv[1:]])


def qhelpconverter():
    load_dotenv()
    return subprocess.call([str(here/'qhelpconverter.exe'), *sys.argv[1:]])


def qhelpgenerator():
    load_dotenv()
    return subprocess.call([str(here/'qhelpgenerator.exe'), *sys.argv[1:]])


def qlalr():
    load_dotenv()
    return subprocess.call([str(here/'qlalr.exe'), *sys.argv[1:]])


def qml():
    load_dotenv()
    return subprocess.call([str(here/'qml.exe'), *sys.argv[1:]])


def qmlcachegen():
    load_dotenv()
    return subprocess.call([str(here/'qmlcachegen.exe'), *sys.argv[1:]])


def qmleasing():
    load_dotenv()
    return subprocess.call([str(here/'qmleasing.exe'), *sys.argv[1:]])


def qmlimportscanner():
    load_dotenv()
    return subprocess.call([str(here/'qmlimportscanner.exe'), *sys.argv[1:]])


def qmllint():
    load_dotenv()
    return subprocess.call([str(here/'qmllint.exe'), *sys.argv[1:]])


def qmlmin():
    load_dotenv()
    return subprocess.call([str(here/'qmlmin.exe'), *sys.argv[1:]])


def qmlplugindump():
    load_dotenv()
    return subprocess.call([str(here/'qmlplugindump.exe'), *sys.argv[1:]])


def qmlprofiler():
    load_dotenv()
    return subprocess.call([str(here/'qmlprofiler.exe'), *sys.argv[1:]])


def qmlscene():
    load_dotenv()
    return subprocess.call([str(here/'qmlscene.exe'), *sys.argv[1:]])


def qmltestrunner():
    load_dotenv()
    return subprocess.call([str(here/'qmltestrunner.exe'), *sys.argv[1:]])


def qscxmlc():
    load_dotenv()
    return subprocess.call([str(here/'qscxmlc.exe'), *sys.argv[1:]])


def qtattributionsscanner():
    load_dotenv()
    return subprocess.call([str(here/'qtattributionsscanner.exe'), *sys.argv[1:]])


def qtdiag():
    load_dotenv()
    return subprocess.call([str(here/'qtdiag.exe'), *sys.argv[1:]])


def qtpaths():
    load_dotenv()
    return subprocess.call([str(here/'qtpaths.exe'), *sys.argv[1:]])


def qtplugininfo():
    load_dotenv()
    return subprocess.call([str(here/'qtplugininfo.exe'), *sys.argv[1:]])


def qvkgen():
    load_dotenv()
    return subprocess.call([str(here/'qvkgen.exe'), *sys.argv[1:]])


def repc():
    load_dotenv()
    return subprocess.call([str(here/'repc.exe'), *sys.argv[1:]])


def testcon():
    load_dotenv()
    return subprocess.call([str(here/'testcon.exe'), *sys.argv[1:]])


def uic():
    load_dotenv()
    return subprocess.call([str(here/'uic.exe'), *sys.argv[1:]])


def xmlpatterns():
    load_dotenv()
    return subprocess.call([str(here/'xmlpatterns.exe'), *sys.argv[1:]])


def xmlpatternsvalidator():
    load_dotenv()
    return subprocess.call([str(here/'xmlpatternsvalidator.exe'), *sys.argv[1:]])


