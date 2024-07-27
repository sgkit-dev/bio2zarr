import cyvcf2
import numpy as np
import numpy.testing as nt
import tqdm
import zarr

from .. import constants


def assert_all_missing_float(a):
    v = np.array(a, dtype=np.float32).view(np.int32)
    nt.assert_equal(v, constants.FLOAT32_MISSING_AS_INT32)


def assert_all_fill_float(a):
    v = np.array(a, dtype=np.float32).view(np.int32)
    nt.assert_equal(v, constants.FLOAT32_FILL_AS_INT32)


def assert_all_missing_int(a):
    v = np.array(a, dtype=int)
    nt.assert_equal(v, constants.INT_MISSING)


def assert_all_fill_int(a):
    v = np.array(a, dtype=int)
    nt.assert_equal(v, constants.INT_FILL)


def assert_all_missing_string(a):
    nt.assert_equal(a, constants.STR_MISSING)


def assert_all_fill_string(a):
    nt.assert_equal(a, constants.STR_FILL)


def assert_all_fill(zarr_val, vcf_type):
    if vcf_type == "Integer":
        assert_all_fill_int(zarr_val)
    elif vcf_type in ("String", "Character"):
        assert_all_fill_string(zarr_val)
    elif vcf_type == "Float":
        assert_all_fill_float(zarr_val)
    else:  # pragma: no cover
        assert False  # noqa PT015


def assert_all_missing(zarr_val, vcf_type):
    if vcf_type == "Integer":
        assert_all_missing_int(zarr_val)
    elif vcf_type in ("String", "Character"):
        assert_all_missing_string(zarr_val)
    elif vcf_type == "Flag":
        assert zarr_val == False  # noqa 712
    elif vcf_type == "Float":
        assert_all_missing_float(zarr_val)
    else:  # pragma: no cover
        assert False  # noqa PT015


def assert_info_val_missing(zarr_val, vcf_type):
    assert_all_missing(zarr_val, vcf_type)


def assert_format_val_missing(zarr_val, vcf_type):
    assert_info_val_missing(zarr_val, vcf_type)


# Note: checking exact equality may prove problematic here
# but we should be deterministically storing what cyvcf2
# provides, which should compare equal.


def assert_info_val_equal(vcf_val, zarr_val, vcf_type):
    assert vcf_val is not None
    if vcf_type in ("String", "Character"):
        split = list(vcf_val.split(","))
        k = len(split)
        if isinstance(zarr_val, str):
            assert k == 1
            # Scalar
            assert vcf_val == zarr_val
        else:
            nt.assert_equal(split, zarr_val[:k])
            assert_all_fill(zarr_val[k:], vcf_type)

    elif isinstance(vcf_val, tuple):
        vcf_missing_value_map = {
            "Integer": constants.INT_MISSING,
            "Float": constants.FLOAT32_MISSING,
        }
        v = [vcf_missing_value_map[vcf_type] if x is None else x for x in vcf_val]
        missing = np.array([j for j, x in enumerate(vcf_val) if x is None], dtype=int)
        a = np.array(v)
        k = len(a)
        # We are checking for int missing twice here, but it's necessary to have
        # a separate check for floats because different NaNs compare equal
        nt.assert_equal(a, zarr_val[:k])
        assert_all_missing(zarr_val[missing], vcf_type)
        if k < len(zarr_val):
            assert_all_fill(zarr_val[k:], vcf_type)
    else:
        # Scalar
        zarr_val = np.array(zarr_val, ndmin=1)
        assert len(zarr_val.shape) == 1
        assert vcf_val == zarr_val[0]
        if len(zarr_val) > 1:
            assert_all_fill(zarr_val[1:], vcf_type)


def assert_format_val_equal(vcf_val, zarr_val, vcf_type, vcf_number):
    assert vcf_val is not None
    assert isinstance(vcf_val, np.ndarray)
    if vcf_type in ("String", "Character"):
        assert len(vcf_val) == len(zarr_val)
        for v, z in zip(vcf_val, zarr_val):
            if vcf_number == "1":
                assert v == z
            else:
                split = list(v.split(","))
                k = len(split)
                nt.assert_equal(split, z[:k])
                assert_all_fill(z[k:], vcf_type)
    else:
        assert vcf_val.shape[0] == zarr_val.shape[0]
        if len(vcf_val.shape) == len(zarr_val.shape) + 1:
            assert vcf_val.shape[-1] == 1
            vcf_val = vcf_val[..., 0]
        assert len(vcf_val.shape) <= 2
        assert len(vcf_val.shape) == len(zarr_val.shape)
        if len(vcf_val.shape) == 2:
            k = vcf_val.shape[1]
            if zarr_val.shape[1] != k:
                assert_all_fill(zarr_val[:, k:], vcf_type)
                zarr_val = zarr_val[:, :k]
        assert vcf_val.shape == zarr_val.shape
        if vcf_type == "Integer":
            vcf_val[vcf_val == constants.VCF_INT_MISSING] = constants.INT_MISSING
            vcf_val[vcf_val == constants.VCF_INT_FILL] = constants.INT_FILL
        elif vcf_type == "Float":
            nt.assert_equal(vcf_val.view(np.int32), zarr_val.view(np.int32))

        nt.assert_equal(vcf_val, zarr_val)


def verify(vcf_path, zarr_path, show_progress=False):
    store = zarr.DirectoryStore(zarr_path)

    root = zarr.group(store=store)
    pos = root["variant_position"][:]
    allele = root["variant_allele"][:]
    chrom = root["contig_id"][:][root["variant_contig"][:]]
    vid = root["variant_id"][:]
    call_genotype = None
    if "call_genotype" in root and root["call_genotype"].size > 0:
        call_genotype = iter(root["call_genotype"])

    vcf = cyvcf2.VCF(vcf_path)
    format_headers = {}
    info_headers = {}
    for h in vcf.header_iter():
        if h["HeaderType"] == "FORMAT":
            format_headers[h["ID"]] = h
        if h["HeaderType"] == "INFO":
            info_headers[h["ID"]] = h

    format_fields = {}
    info_fields = {}
    for colname in root.keys():
        if colname.startswith("call") and not colname.startswith("call_genotype"):
            vcf_name = colname.split("_", 1)[1]
            vcf_type = format_headers[vcf_name]["Type"]
            vcf_number = format_headers[vcf_name]["Number"]
            format_fields[vcf_name] = vcf_type, vcf_number, iter(root[colname])
        if colname.startswith("variant"):
            name = colname.split("_", 1)[1]
            if name.isupper():
                vcf_type = info_headers[name]["Type"]
                info_fields[name] = vcf_type, iter(root[colname])

    first_pos = next(vcf).POS
    start_index = np.searchsorted(pos, first_pos)
    assert pos[start_index] == first_pos
    vcf = cyvcf2.VCF(vcf_path)
    if show_progress:
        iterator = tqdm.tqdm(vcf, desc="  Verify", total=vcf.num_records)  # NEEDS TEST
    else:
        iterator = vcf
    for j, row in enumerate(iterator, start_index):
        assert chrom[j] == row.CHROM
        assert pos[j] == row.POS
        assert vid[j] == ("." if row.ID is None else row.ID)
        assert allele[j, 0] == row.REF
        k = len(row.ALT)
        nt.assert_array_equal(allele[j, 1 : k + 1], row.ALT)
        assert np.all(allele[j, k + 1 :] == "")
        # TODO FILTERS

        if call_genotype is None:
            val = None
            try:
                val = row.format("GT")
            except KeyError:
                pass
            assert val is None
        else:
            gt = row.genotype.array()
            gt_zarr = next(call_genotype)
            gt_vcf = gt[:, :-1]
            # NOTE cyvcf2 remaps genotypes automatically
            # into the same missing/pad encoding that sgkit uses.
            nt.assert_array_equal(gt_zarr, gt_vcf)

        for name, (vcf_type, zarr_iter) in info_fields.items():
            vcf_val = row.INFO.get(name, None)
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                assert_info_val_missing(zarr_val, vcf_type)
            else:
                assert_info_val_equal(vcf_val, zarr_val, vcf_type)

        for name, (vcf_type, vcf_number, zarr_iter) in format_fields.items():
            vcf_val = row.format(name)
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                assert_format_val_missing(zarr_val, vcf_type)
            else:
                assert_format_val_equal(vcf_val, zarr_val, vcf_type, vcf_number)
