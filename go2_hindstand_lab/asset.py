"""Prepare the existing URDF for Lab, folding only massless fixed links."""
from copy import deepcopy
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "resources/robots/go2/urdf/go2.urdf"
JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
               for part in ("hip", "thigh", "calf")]
FOOT_NAMES = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]


def transform(element):
    origin = element.find("origin")
    matrix = np.eye(4)
    if origin is not None:
        matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return matrix


def set_transform(element, matrix):
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    origin.set("xyz", " ".join(format(v, ".16g") for v in matrix[:3, 3]))
    origin.set("rpy", " ".join(format(v, ".16g") for v in Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")))


def prepare_urdf():
    robot = ET.parse(SOURCE).getroot()
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", str((SOURCE.parent / mesh.get("filename")).resolve()))
    for joint in list(robot.findall("joint")):
        if joint.get("type") != "fixed":
            continue
        links = {link.get("name"): link for link in robot.findall("link")}
        child_name = joint.find("child").get("link")
        child = links[child_name]
        mass = child.find("inertial/mass")
        if mass is not None and float(mass.get("value")) > 0:
            continue
        parent_name = joint.find("parent").get("link")
        parent = links[parent_name]
        pose = transform(joint)
        for kind in ("visual", "collision"):
            for item in child.findall(kind):
                item = deepcopy(item)
                set_transform(item, pose @ transform(item))
                parent.append(item)
        for descendant in robot.findall("joint"):
            if descendant.find("parent").get("link") == child_name:
                set_transform(descendant, pose @ transform(descendant))
                descendant.find("parent").set("link", parent_name)
        robot.remove(joint)
        robot.remove(child)
    target = ROOT / ".cache/isaaclab/go2.urdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    content = ET.tostring(robot, encoding="utf-8", xml_declaration=True)
    if not target.exists() or target.read_bytes() != content:
        target.write_bytes(content)
    return target


def asset_metadata():
    robot = ET.parse(SOURCE).getroot()
    limits = {j.get("name"): j.find("limit").attrib for j in robot.findall("joint") if j.get("type") == "revolute"}
    return {"urdf_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "mass": sum(float(m.get("value")) for m in robot.findall("link/inertial/mass")),
            "limits": limits}
