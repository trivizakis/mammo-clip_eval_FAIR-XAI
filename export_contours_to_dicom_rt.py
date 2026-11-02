import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def export_contours_to_dicom_rt(
    reference_dcm_path: str,
    contour_results: dict,
    output_rt_path: str
):
    """
    Creates a valid DICOM-RT Structure Set (IOD) from pixel contours
    projected into patient physical coordinate space.
    """
    # 1. Load reference mammogram DICOM (metadata only)
    ref_ds = pydicom.dcmread(reference_dcm_path, stop_before_pixels=True)
    raw_spacing = getattr(ref_ds, "PixelSpacing", None) or getattr(ref_ds, "ImagerPixelSpacing", None) or [0.1, 0.1]
    pixel_spacing = [float(x) for x in raw_spacing]
    raw_pos = getattr(ref_ds, "ImagePositionPatient", None) or [0.0, 0.0, 0.0]
    img_pos = [float(x) for x in raw_pos]

    # 2. File Meta Information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    rt_ds = FileDataset(output_rt_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # 3. Patient & Study Attributes
    rt_ds.PatientName = getattr(ref_ds, "PatientName", "ANONYMOUS")
    rt_ds.PatientID = getattr(ref_ds, "PatientID", "UNKNOWN")
    rt_ds.StudyInstanceUID = getattr(ref_ds, "StudyInstanceUID", generate_uid())
    rt_ds.SeriesInstanceUID = generate_uid()
    rt_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    rt_ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ref_sop_class = getattr(ref_ds, "SOPClassUID", None) or getattr(getattr(ref_ds, "file_meta", None), "MediaStorageSOPClassUID", "1.2.840.10008.5.1.4.1.1.1.2")
    ref_sop_instance = getattr(ref_ds, "SOPInstanceUID", None) or getattr(getattr(ref_ds, "file_meta", None), "MediaStorageSOPInstanceUID", generate_uid())

    rt_ds.Modality = "RTSTRUCT"
    rt_ds.StructureSetLabel = "MAMMO_XAI_RT"

    now = datetime.datetime.now()
    rt_ds.StructureSetDate = now.strftime("%Y%m%d")
    rt_ds.StructureSetTime = now.strftime("%H%M%S")

    # 4. Link Reference Frame of Reference & SOP Instance
    frame_uid = getattr(ref_ds, "FrameOfReferenceUID", generate_uid())
    rt_ds.ReferencedFrameOfReferenceSequence = Sequence()
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = frame_uid
    ref_for.RTReferencedStudySequence = Sequence()

    ref_study = Dataset()
    ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    ref_study.ReferencedSOPInstanceUID = rt_ds.StudyInstanceUID
    ref_study.RTReferencedSeriesSequence = Sequence()

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = getattr(ref_ds, "SeriesInstanceUID", generate_uid())
    ref_series.ContourImageSequence = Sequence()

    contour_img = Dataset()
    contour_img.ReferencedSOPClassUID = ref_sop_class
    contour_img.ReferencedSOPInstanceUID = ref_sop_instance
    ref_series.ContourImageSequence.append(contour_img)

    ref_study.RTReferencedSeriesSequence.append(ref_series)
    ref_for.RTReferencedStudySequence.append(ref_study)
    rt_ds.ReferencedFrameOfReferenceSequence.append(ref_for)

    # 5. Populate Structure Set ROI & ROI Contour Sequences
    rt_ds.StructureSetROISequence = Sequence()
    rt_ds.ROIContourSequence = Sequence()

    roi_specs = [
        ("Core_Nidus", [255, 0, 0], contour_results.get("core_contours", [])),        # Red
        ("Peritumoral_Margin", [0, 255, 255], contour_results.get("margin_contours", [])) # Cyan
    ]

    for roi_num, (roi_name, color, contours) in enumerate(roi_specs, start=1):
        # Structure Set ROI entry
        ss_roi = Dataset()
        ss_roi.ROINumber = roi_num
        ss_roi.ReferencedFrameOfReferenceUID = frame_uid
        ss_roi.ROIName = roi_name
        ss_roi.ROIGenerationAlgorithm = "AUTOMATIC"
        rt_ds.StructureSetROISequence.append(ss_roi)

        # ROI Contour entry
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = roi_num
        roi_contour.ROIDisplayColor = color
        roi_contour.ContourSequence = Sequence()

        for c in contours:
            pts = c.squeeze()
            if len(pts) < 3:
                continue

            # Map (px_x, px_y) -> Physical patient space (X, Y, Z)
            # X = X_orig + col * spacing_x; Y = Y_orig + row * spacing_y
            coord_data = []
            for px, py in pts:
                world_x = img_pos[0] + px * pixel_spacing[1]
                world_y = img_pos[1] + py * pixel_spacing[0]
                world_z = img_pos[2]
                coord_data.extend([world_x, world_y, world_z])

            c_dataset = Dataset()
            c_dataset.ContourGeometricType = "CLOSED_PLANAR"
            c_dataset.NumberOfContourPoints = len(pts)
            c_dataset.ContourData = coord_data

            img_ref_seq = Sequence()
            single_img_ref = Dataset()
            single_img_ref.ReferencedSOPClassUID = ref_sop_class
            single_img_ref.ReferencedSOPInstanceUID = ref_sop_instance
            img_ref_seq.append(single_img_ref)
            c_dataset.ContourImageSequence = img_ref_seq

            roi_contour.ContourSequence.append(c_dataset)

        rt_ds.ROIContourSequence.append(roi_contour)

    rt_ds.is_little_endian = True
    rt_ds.is_implicit_VR = False
    rt_ds.save_as(output_rt_path, write_like_original=False)
    print(f"DICOM-RT Structure Set exported to {output_rt_path}")