"""The few checks that guard what matters: output pixels, saved work, and
basic detection. Everything else is quicker to notice by using the tool."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import scan_cropper_auto as core
import scan_cropper_gui as gui
from scan_cropper import (render_transformed, rotated_full_box, save_crop,
                          transformed_crop, transformed_crop_with_channel_shifts)
from scan_cropper_auto import analyze_array, analyze_batch, rotated, solve_batch

SHIFTS={'red':(.081,-.052),'green':(0.,0.),'blue':(.966,.123)}


def synthetic(angle=0.):
    """Dark bed, a page with a dark printed margin, tilted clockwise by angle."""
    rng=np.random.default_rng(21)
    a=np.clip(rng.normal(35,4,(1100,900)),0,255).astype(np.uint8)
    a[60:1030,125:800]=85
    a[60:1030,185:740]=220
    cv2.rectangle(a,(300,220),(550,750),70,-1)
    if angle:
        a=cv2.warpAffine(a,cv2.getRotationMatrix2D((450,550),-angle,1),(900,1100),borderValue=35)
    return a


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


# --- Detection ---------------------------------------------------------------

def test_detection_finds_paper_edges_and_clockwise_tilt():
    page=analyze_array(synthetic(.55))
    assert abs(page.angles['left']+.55)<.09 and page.confidence['left']>.7
    assert 45<np.median(np.asarray(page.edges['top'])[:,1])<75
    # The physical page (~675 px), not the narrower light printed area.
    width=np.median(np.asarray(page.edges['right'])[:,0])-np.median(np.asarray(page.edges['left'])[:,0])
    assert 660<width<690


def test_batch_size_fits_pages_and_ignores_one_unusual_page():
    from dataclasses import replace
    pages=[analyze_array(synthetic(a)) for a in (0.,.2,-.3)]
    size,plans=solve_batch(pages,margins=(3,3,3,6))
    assert 640<size[0]<680 and 930<size[1]<975
    assert all((p.box[2]-p.box[0],p.box[3]-p.box[1])==size for p in plans)
    base=analyze_array(synthetic())
    batch=[replace(base,name=f'{i}.png') for i in range(8)]
    short=dict(base.edges); short['bottom']=[[x,y-120] for x,y in base.edges['bottom']]
    batch[3]=replace(base,name='3.png',edges=short)
    size,_=solve_batch(batch,mode='left',margins=(0,0,0,0))
    assert abs(size[1]-core_height(base))<=8


def core_height(page):
    return np.median(np.asarray(page.edges['bottom'])[:,1])-np.median(np.asarray(page.edges['top'])[:,1])


def bed_with_page(band_rows=0):
    """Grainy bed and a page whose bottom is printed dark and smooth."""
    rng=np.random.default_rng(5)
    a=np.clip(rng.normal(60,12,(1400,1000)),0,255).astype(np.uint8)
    a[150:1250,150:850]=225
    if band_rows:
        a[1250-band_rows:1250,150:850]=np.clip(rng.normal(62,2,(band_rows,700)),0,255).astype(np.uint8)
    return a


def test_scanner_adapter_recovers_hidden_edge_and_is_optional(tmp_path,monkeypatch):
    paths=[]
    for i in range(6):
        path=tmp_path/f'{i:02}.png'
        Image.fromarray(bed_with_page(40) if i==3 else bed_with_page()).save(path)
        paths.append(path)
    bottom=lambda pages:np.median(np.asarray(pages[3].edges['bottom'])[:,1])
    assert abs(bottom(analyze_batch(paths))-1250)<=6
    # Without scan_cropper_adapters.py the generic detection still works.
    monkeypatch.setitem(sys.modules,'scan_cropper_adapters',None)
    monkeypatch.setattr(core,'_ADAPTERS',None)
    assert core.adapters() is None
    assert abs(bottom(analyze_batch(paths))-1210)<=6
    monkeypatch.setattr(gui.CropperWindow,'_persist_color_profiles',lambda self:True)
    QApplication.instance() or QApplication([])
    gui.CropperWindow().close()


# --- Output pixels -----------------------------------------------------------

def test_export_rotation_maps_source_geometry():
    a=np.zeros((160,200),np.uint8); a[49:52,119:122]=255
    angle=.8; box=(20,10,180,150)
    target=rotated([(120,50)],angle,200,160)[0]-np.array(box[:2])
    out=np.asarray(transformed_crop(Image.fromarray(a),box,angle,(100,80)))
    yy,xx=np.unravel_index(out.argmax(),out.shape)
    assert np.linalg.norm(np.array([xx,yy])-target)<2


@pytest.mark.parametrize('mode,shifts',[('RGB',SHIFTS),('L',None)])
def test_parallel_render_is_bit_identical(mode,shifts):
    rng=np.random.default_rng(3)
    shape=(613,401,3) if mode=='RGB' else (613,401)
    image=Image.fromarray(rng.integers(0,256,shape,dtype=np.uint8),mode)
    center=(image.width/2,image.height/2)
    perspective=np.array([[1.002,.003,-4],[-.002,.999,6],[1e-6,-2e-6,1]])
    for angle,matrix in ((.37,None),(-1.3,perspective)):
        box=rotated_full_box(image.width,image.height,angle,center)
        with ThreadPoolExecutor(4) as executor:
            fast=render_transformed(image,box,angle,center,shifts,matrix,executor,stripe_height=50)
        exact=transformed_crop_with_channel_shifts(image,box,angle,center,shifts,matrix)
        assert fast.mode==exact.mode and np.array_equal(np.asarray(fast),np.asarray(exact))


def test_preview_pixels_match_export(app,tmp_path):
    pixels=np.zeros((180,240,3),np.uint8)
    pixels[20:160,30:210]=230
    pixels[40:130:7,40:200]=[200,80,40]
    path=tmp_path/'page.png'; Image.fromarray(pixels).save(path)
    angle,center=.7,(120,90)
    worker=gui.FullPreviewWorker(1,[(path,angle,center,None,SHIFTS)],threading.Event())
    frames=[]; worker.signals.ready.connect(frames.append)
    worker.run()
    _,_,box,preview=frames[0]
    target=tmp_path/'output.png'
    save_crop(path,target,box,angle,center,SHIFTS)
    assert np.array_equal(np.asarray(preview),np.asarray(Image.open(target)))


# --- Saved work --------------------------------------------------------------

def test_batch_config_roundtrip_keeps_every_crop(app,tmp_path,monkeypatch):
    monkeypatch.setattr(gui.CropperWindow,'_persist_color_profiles',lambda self:True)
    paths=[tmp_path/'003.png',tmp_path/'004.png']
    for path in paths:
        Image.fromarray(synthetic(.2)).save(path)
    pages=[analyze_array(synthetic(.2),p.name) for p in paths]
    size,plans=solve_batch(pages,margins=(0,0,0,0))
    w=gui.CropperWindow()
    monkeypatch.setattr(w,'load_current_preview',lambda:None)
    w.smart_is_new=True; w.smart_finished((paths,pages,size,plans))
    w.crop_outer_spin.setValue(22); w.offset_x_spin.setValue(13); w.rotation_spin.setValue(.31)
    before=w.current_geometry(); rotations=dict(w.page_rotations)
    target=tmp_path/'batch.json'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(target),''))
    w.export_batch_config()
    w.crop_outer_spin.setValue(33); w.offset_x_spin.setValue(-5)
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(target),''))
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.StandardButton.Yes)
    w.import_batch_config()
    assert w.current_geometry()==before and w.page_rotations==rotations
    w.close()
