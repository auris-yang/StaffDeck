import json

from sqlmodel import Session, SQLModel, create_engine, select

from app.agents.branching import agent_private_metadata, is_open_gallery_resource
from app.db.models import (
    AgentProfile,
    AgentResourceBinding,
    GeneralSkill,
    KnowledgeBase,
    Skill,
    Tool,
)
from app.db.staffdeck_seed import TENANT_ID, seed_staffdeck_admin_gallery
from staffdeck_harness.distribution.bootstrap import configure_environment, initialize_defaults
from staffdeck_harness.distribution.manifest import make_manifest
from staffdeck_harness.distribution.resources import export_local
from staffdeck_harness.distribution.schema import PackageRequest
from staffdeck_harness.modules.config import RuntimeOverrides

RESOURCE_TYPES = ('skill', 'knowledge_base', 'general_skill', 'tool')


def _seeded_session():
    engine = create_engine('sqlite://')
    SQLModel.metadata.create_all(engine)
    db = Session(engine)
    db.add(AgentProfile(id=f'agent_{TENANT_ID}_overall', tenant_id=TENANT_ID,
                        name='整体智能体', description='全局资源池', is_overall=True))
    db.flush()
    seed_staffdeck_admin_gallery(db, include_expanded=False)
    db.flush()
    return engine, db


def _employee_ids(db):
    return [row.id for row in db.exec(select(AgentProfile)).all() if not row.is_overall]


def test_export_local_preserves_open_gallery_bindings():
    engine, db = _seeded_session()
    try:
        overall = db.exec(select(AgentProfile).where(AgentProfile.is_overall.is_(True))).one()
        graph = export_local(db, TENANT_ID, _employee_ids(db))
        assert graph['overall_agent_id'] == overall.id
        exported = graph['records']['AgentResourceBinding']
        gallery = [row for row in exported if row['agent_id'] == overall.id]
        assert {row['resource_type'] for row in gallery} == set(RESOURCE_TYPES)
        assert all(row['metadata_json']['visibility'] == 'open_gallery' for row in gallery)
        assert not any(row['id'] == overall.id for row in graph['records']['AgentProfile'])
    finally:
        db.close()
        engine.dispose()


def test_export_local_skips_private_and_deleted_overall_bindings():
    engine, db = _seeded_session()
    try:
        overall = db.exec(select(AgentProfile).where(AgentProfile.is_overall.is_(True))).one()
        private_tool = Tool(tenant_id=TENANT_ID, name='私有工具', method='POST',
                            url='https://example.invalid/private')
        db.add(private_tool)
        db.flush()
        db.add(AgentResourceBinding(
            tenant_id=TENANT_ID, agent_id=overall.id, resource_type='tool',
            resource_id=private_tool.id, status='active',
            metadata_json=agent_private_metadata(overall.id)))
        deleted_binding = db.exec(select(AgentResourceBinding).where(
            AgentResourceBinding.agent_id == overall.id)).first()
        deleted_binding.status = 'deleted'
        db.add(deleted_binding)
        db.flush()
        graph = export_local(db, TENANT_ID, _employee_ids(db))
        exported_ids = {row['id'] for row in graph['records']['AgentResourceBinding']}
        assert not any(row['resource_id'] == private_tool.id and row['agent_id'] == overall.id
                       for row in graph['records']['AgentResourceBinding'])
        assert deleted_binding.id not in exported_ids
    finally:
        db.close()
        engine.dispose()


def test_explicit_export_install_remaps_gallery_bindings_to_local_overall(tmp_path, monkeypatch):
    engine, db = _seeded_session()
    try:
        graph = export_local(db, TENANT_ID, _employee_ids(db))
    finally:
        db.close()
        engine.dispose()

    manifest = make_manifest(PackageRequest(tenant_id='fresh', kind='source'), RuntimeOverrides())
    manifest['resources'] = graph
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setenv('STAFFDECK_DISTRIBUTION_MANIFEST', str(path))
    monkeypatch.setenv('ULTRARAG_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('ULTRARAG_DOTENV', str(tmp_path / 'missing.env'))
    monkeypatch.delenv('STAFFDECK_INITIAL_ADMIN_PASSWORD', raising=False)
    monkeypatch.delenv('APP_SECRET', raising=False)
    monkeypatch.delenv('DATABASE_URL', raising=False)
    configure_environment()

    target = create_engine('sqlite://')
    SQLModel.metadata.create_all(target)
    try:
        with Session(target) as db:
            assert initialize_defaults(db, manifest)
            overall = db.exec(select(AgentProfile).where(AgentProfile.is_overall.is_(True))).one()
            assert overall.id == 'agent_fresh_overall'
            assert len(db.exec(select(AgentProfile).where(AgentProfile.is_overall.is_(False))).all()) == 5
            for resource_type, model in (('skill', Skill), ('knowledge_base', KnowledgeBase),
                                         ('general_skill', GeneralSkill), ('tool', Tool)):
                bindings = db.exec(select(AgentResourceBinding).where(
                    AgentResourceBinding.agent_id == overall.id,
                    AgentResourceBinding.resource_type == resource_type)).all()
                assert bindings
                assert all(row.tenant_id == 'fresh' for row in bindings)
                for binding in bindings:
                    resource = db.get(model, binding.resource_id)
                    assert resource is not None and resource.tenant_id == 'fresh'
                    assert is_open_gallery_resource(db, 'fresh', resource_type, resource)
            assert not db.exec(select(AgentResourceBinding).where(
                AgentResourceBinding.agent_id == graph['overall_agent_id'])).all()
    finally:
        target.dispose()