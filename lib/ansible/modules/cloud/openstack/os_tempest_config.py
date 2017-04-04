#!/usr/bin/python

ANSIBLE_METADATA = {'metadata_version': '1.0',
                    'status': ['preview'],
                    'supported_by': 'community'}

DOCUMENTATION = '''
---
module: os_tempest_config
short_description: configs Tempest (OpenStack)
description: derived from Red Hat's config_tempest.py script and Red Hat's Tempest api_discovery.py script
    -
author: "Tal Shafir , @TalShafir"
requirements: ["tempest", "urllib3 >= 1.15", "requests"]
    -
options:
    overrides_file:
        description:
            path for a file containing overrides values in the format section.option=value
            where section is a section header in the configuration file.
        required: False
        default: ''
    deployer_input:
        description:
            path for a file in the format of Tempest's configuration file that will override the default values.
            The deployer-input file is an alternative to providing key/value pairs.
            If there are also key/value pairs they will be applied after the deployer-input file.
        required: False
        default: ''
    defaults_file:
        description:
            a path for a defaults file that will be provided by the distributor of the tempest code,
            a distro for example,to specify defaults that are different than the generic defaults for tempest.
        required: False
        default: ''
    overrides:
        description: 
            key value pairs to modify. The key is section.key=value where section is a section header in the configuration file.
        required: False
        default: ''
    create:
        description:
            When C(True) create default tempest resources
        required: False
        default: False
    admin_cred:
        description:
            When C(True) Run with admin creds
        required: False
        default: False
    use_test_accounts:
        description:
            When C(True) use accounts from accounts.yaml
        required: False
        default: False
    image_disk_format:
        description:
            a format of an image to be uploaded to glance
        required: False
        default: 'qcow2'
    image:
        description:
            an image to be uploaded to glance. The name of the image is the leaf name of the path which can be either a filename or url.
        required: False
        default: "http://download.cirros-cloud.net/0.3.4/cirros-0.3.4-x86_64-disk.img"
    network_id:
        description:
            The ID of an existing network in our OpenStack instance with external connectivity
        required: False
        default: ''
    log_file:
        description:
            If given  will write all the log info to the given file.
        required: False
        default: os.devnull
'''

EXAMPLES = '''

'''

from ansible.module_utils.basic import AnsibleModule

import ConfigParser
import logging
import os
import shutil

import ansible.module_utils.urls

try:
    import tempest.config
    from tempest.lib import auth
    from tempest.lib import exceptions
    from tempest.lib.services.compute import flavors_client
    from tempest.lib.services.compute import networks_client as nova_net_client
    from tempest.lib.services.compute import servers_client
    from tempest.lib.services.identity.v2 import identity_client
    from tempest.lib.services.identity.v2 import roles_client
    from tempest.lib.services.identity.v2 import tenants_client
    from tempest.lib.services.identity.v2 import users_client
    from tempest.lib.services.identity.v3 \
        import identity_client as identity_v3_client
    from tempest.lib.services.image.v2 import images_client
    from tempest.lib.services.network import networks_client
    from tempest.common import identity

    HAS_TEMPEST = True
except ImportError:
    HAS_TEMPEST = False

import json
import re
import urlparse

try:
    import urllib3
    import requests

    HAS_URL = True
except ImportError:
    HAS_URL = True

# imports for the code copied from ansible.utils.path
from errno import EEXIST
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text


LOG = logging.getLogger(__name__)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

DEFAULT_IMAGE = ("http://download.cirros-cloud.net/0.3.4/"
                 "cirros-0.3.4-x86_64-disk.img")
DEFAULT_IMAGE_FORMAT = 'qcow2'

# services and their codenames
SERVICE_NAMES = {
    'baremetal': 'ironic',
    'compute': 'nova',
    'database': 'trove',
    'data-processing': 'sahara',
    'image': 'glance',
    'network': 'neutron',
    'object-store': 'swift',
    'orchestration': 'heat',
    'telemetry': 'ceilometer',
    'volume': 'cinder',
    'messaging': 'zaqar',
}

# what API versions could the service have and should be enabled/disabled
# depending on whether they get discovered as supported. Services with only one
# version don't need to be here, neither do service versions that are not
# configurable in tempest.conf
SERVICE_VERSIONS = {
    'image': ['v1', 'v2'],
    'identity': ['v2', 'v3'],
    'volume': ['v1', 'v2']
}

# Keep track of where the extensions are saved for that service.
# This is necessary because the configuration file is inconsistent - it uses
# different option names for service extension depending on the service.
SERVICE_EXTENSION_KEY = {
    'compute': 'api_extensions',
    'object-store': 'discoverable_apis',
    'network': 'api_extensions',
    'volume': 'api_extensions',
    'identity': 'api_extensions'
}


# TODO change overrides argument to be string in a format

def main():
    module = AnsibleModule(argument_spec={
        "output_path": {"type": "path", "required": True},
        "overrides_file": {"type": "path", "required": False, "default": ""},
        "defaults_file": {"type": "path", "required": False, "default": ""},
        "deployer_input": {"type": "path", "required": False, "default": ""},
        "overrides": {"type": "list", "required": False, "default": ""},
        "create": {"type": "bool", "required": False, "default": False},
        "admin_cred": {"type": "bool", "required": False, "default": False},
        "use_test_accounts": {"type": "bool", "required": False, "default": False},
        "image_disk_format": {"type": "str", "required": False, "default": DEFAULT_IMAGE_FORMAT},
        "image": {"type": "str", "required": False, "default": DEFAULT_IMAGE},
        "network_id": {"type": "str", "required": False, "default": ""},
        "log_file": {"type": "path", "required": False, "default": ""},
    })
    if module.params["create"] and not module.params["admin_cred"]:
        module.fail_json(msg="Cannot use 'create' param without 'admin_cred' param as True")
    if module.params["deployer_input"] and not os.path.isfile(unfrackpath(module.params["deployer_input"])):
        module.fail_json(msg="the deployer_input file is not a file")

    if not HAS_TEMPEST:
        module.fail_json(msg="Failed to import Tempest")
    if not HAS_URL:
        module.fail_json(msg="Failed to import urllib3 or requests")

    try:
        log_file_path = unfrackpath(module.params["log_file"])
        prepare_path(log_file_path)
        conf = TempestConf(log_file_path)

        if module.params["defaults_file"] and os.path.isfile(module.params["defaults_file"]):
            LOG.info("Reading defaults from file '%s'", module.params["defaults_file"])
            conf.read(module.params["defaults_file"])
        if module.params["deployer_input"] and os.path.isfile(module.params["deployer_input"]):
            LOG.info("Adding options from deployer-input file '%s'",
                     module.params["deployer_input"])
            deployer_input = ConfigParser.SafeConfigParser()
            deployer_input.read(module.params["deployer_input"])
            for section in deployer_input.sections():
                # There are no deployer input options in DEFAULT
                for (key, value) in deployer_input.items(section):
                    conf.set(section, key, value, priority=True)
        if module.params["overrides"]:
            for section, key, value in parse_overrides(module.params["overrides"]):
                conf.set(section, key, value, priority=True)

        # validate necessary settings are being given
        if module.params["admin_cred"]:
            if not conf.has_option("identity", "admin_username"):
                module.fail_json(
                    msg="the 'admin_cred' option require 'admin_username' to be provided by the user or as a default")
            if not conf.has_option("identity", "admin_tenant_name"):
                module.fail_json(
                    msg="the 'admin_cred' option require 'admin_tenant_name' to be provided by the user or as a default")
            if not conf.has_option("identity", "admin_password"):
                module.fail_json(
                    msg="the 'admin_cred' option require 'admin_password' to be provided by the user or as a default")
        if not conf.has_option("identity", "uri"):
            module.fail_json(msg="the identity.uri setting must be provided")

        uri = conf.get("identity", "uri")
        api_version = 2
        v3_only = False
        if "v3" in uri and v3_only:
            api_version = 3
        if "v3" in uri:
            conf.set("identity", "auth_version", "v3")
            conf.set("identity", "uri", uri.replace("v3", "v2.0"), priority=True)
            conf.set("identity", "uri_v3", uri)
        else:
            conf.set("identity", "uri_v3", uri.replace("v2.0", "v3"))
        if not module.params["admin_cred"]:
            conf.set("identity", "admin_username", "")
            conf.set("identity", "admin_tenant_name", "")
            conf.set("identity", "admin_password", "")
            conf.set("auth", "allow_tenant_isolation", "False")
        if module.params["use_test_accounts"]:
            conf.set("auth", "allow_tenant_isolation", "True")

        clients = ClientManager(conf, module.params["admin_cred"])

        swift_discover = conf.get_defaulted('object-storage-feature-enabled',
                                            'discoverability')
        services = discover(
            clients.auth_provider,
            clients.identity_region,
            object_store_discovery=conf.get_bool_value(swift_discover),
            api_version=api_version,
            disable_ssl_certificate_validation=conf.get_defaulted(
                'identity',
                'disable_ssl_certificate_validation'
            )
        )

        if module.params["create"] and not module.params["use_test_accounts"]:
            create_tempest_users(clients.tenants, clients.roles, clients.users,
                                 conf, services)

        create_tempest_flavors(clients.flavors, conf, module.params["create"])
        create_tempest_images(clients.images, conf, module.params["image"], module.params["create"],
                              module.params["image_disk_format"])
        has_neutron = "network" in services

        LOG.info("Setting up network")
        LOG.debug("Is neutron present: {0}".format(has_neutron))
        create_tempest_networks(clients, conf, has_neutron, module.params["network_id"])

        configure_discovered_services(conf, services)
        configure_boto(conf, services)
        configure_horizon(conf)
        LOG.info("Creating configuration file %s" % os.path.abspath(module.params["output_path"]))

        output_path = unfrackpath(module.params["output_path"])
        if os.path.isdir(output_path):
            output_path += "tempest.conf"
        with open(output_path, 'w') as f:
            conf.write(f)

        module.exit_json(msg="generated tempest.conf successfully",
                         config_path=unfrackpath(module.params["output_path"]))
    except Exception as error:
        module.fail_json(msg=str(error))


def prepare_path(file_path):
    """Creates the path to the dir that contains the file if it doesn't exists"""

    dir_name = os.path.dirname(file_path)
    if not os.path.isdir(dir_name):
        makedirs_safe(dir_name)


# copied from ansible.utils.path
def unfrackpath(path, follow=True):
    """
    Returns a path that is free of symlinks (if follow=True), environment variables, relative path traversals and symbols (~)

    :arg path: A byte or text string representing a path to be canonicalized
    :arg follow: A boolean to indicate of symlinks should be resolved or not
    :raises UnicodeDecodeError: If the canonicalized version of the path
        contains non-utf8 byte sequences.
    :rtype: A text string (unicode on pyyhon2, str on python3).
    :returns: An absolute path with symlinks, environment variables, and tilde
        expanded.  Note that this does not check whether a path exists.

    example::
        '$HOME/../../var/mail' becomes '/var/spool/mail'
    """

    if follow:
        final_path = os.path.normpath(
            os.path.realpath(os.path.expanduser(os.path.expandvars(to_bytes(path, errors='surrogate_or_strict')))))
    else:
        final_path = os.path.normpath(
            os.path.abspath(os.path.expanduser(os.path.expandvars(to_bytes(path, errors='surrogate_or_strict')))))

    return to_text(final_path, errors='surrogate_or_strict')

# copied from ansible.utils.path
def makedirs_safe(path, mode=None):
    """Safe way to create dirs in muliprocess/thread environments.

    :arg path: A byte or text string representing a directory to be created
    :kwarg mode: If given, the mode to set the directory to
    :raises Exception: If the directory cannot be created and does not already exists.
    :raises UnicodeDecodeError: if the path is not decodable in the utf-8 encoding.
    """

    rpath = unfrackpath(path)
    b_rpath = to_bytes(rpath)
    if not os.path.exists(b_rpath):
        try:
            if mode:
                os.makedirs(b_rpath, mode)
            else:
                os.makedirs(b_rpath)
        except OSError as e:
            if e.errno != EEXIST:
                raise Exception("Unable to create local directories(%s): %s" % (to_native(rpath), to_native(e)))


def parse_overrides(overrides):
    """Manual parsing of positional arguments.

    TODO(mkollaro) find a way to do it in argparse
    """
    if len(overrides) % 2 != 0:
        raise Exception("An odd number of override options was found. The"
                        " overrides have to be in 'section.key value' format. ")
    i = 0
    new_overrides = []
    while i < len(overrides):
        section_key = overrides[i].split('.')
        value = overrides[i + 1]
        if len(section_key) != 2:
            raise Exception("Missing dot. The option overrides has to come in"
                            " the format 'section.key value', but got '%s'."
                            % (overrides[i] + ' ' + value))
        section, key = section_key
        new_overrides.append((section, key, value))
        i += 2
    return new_overrides


class ClientManager(object):
    """Manager of various OpenStack API clients.

    Connections to clients are created on-demand, i.e. the client tries to
    connect to the server only when it's being requested.
    """

    def get_credentials(self, conf, username, tenant_name, password,
                        identity_version='v2'):
        creds_kwargs = {'username': username,
                        'password': password}
        if identity_version == 'v3':
            creds_kwargs.update({'project_name': tenant_name,
                                 'domain_name': 'Default',
                                 'user_domain_name': 'Default'})
        else:
            creds_kwargs.update({'tenant_name': tenant_name})
        return auth.get_credentials(
            auth_url=None,
            fill_in=False,
            identity_version=identity_version,
            disable_ssl_certificate_validation=conf.get_defaulted(
                'identity',
                'disable_ssl_certificate_validation'),
            ca_certs=conf.get_defaulted(
                'identity',
                'ca_certificates_file'),
            **creds_kwargs)

    def get_auth_provider(self, conf, credentials):
        disable_ssl_certificate_validation = conf.get_defaulted(
            'identity',
            'disable_ssl_certificate_validation')
        ca_certs = conf.get_defaulted(
            'identity',
            'ca_certificates_file')
        if isinstance(credentials, auth.KeystoneV3Credentials):
            return auth.KeystoneV3AuthProvider(
                credentials, conf.get_defaulted('identity', 'uri_v3'),
                disable_ssl_certificate_validation,
                ca_certs)
        else:
            return auth.KeystoneV2AuthProvider(
                credentials, conf.get_defaulted('identity', 'uri'),
                disable_ssl_certificate_validation,
                ca_certs)

    def get_identity_version(self, conf):
        if "v3" in conf.get("identity", "uri"):
            return "v3"
        else:
            return "v2"

    def __init__(self, conf, admin):

        self.identity_version = self.get_identity_version(conf)
        if admin:
            username = conf.get_defaulted('identity', 'admin_username')
            password = conf.get_defaulted('identity', 'admin_password')
            tenant_name = conf.get_defaulted('identity', 'admin_tenant_name')
        else:
            username = conf.get_defaulted('identity', 'username')
            password = conf.get_defaulted('identity', 'password')
            tenant_name = conf.get_defaulted('identity', 'tenant_name')

        self.identity_region = conf.get_defaulted('identity', 'region')
        default_params = {
            'disable_ssl_certificate_validation':
                conf.get_defaulted('identity',
                                   'disable_ssl_certificate_validation'),
            'ca_certs': conf.get_defaulted('identity', 'ca_certificates_file')
        }
        compute_params = {
            'service': conf.get_defaulted('compute', 'catalog_type'),
            'region': self.identity_region,
            'endpoint_type': conf.get_defaulted('compute', 'endpoint_type')
        }
        compute_params.update(default_params)

        if self.identity_version == "v2":
            _creds = self.get_credentials(conf, username, tenant_name,
                                          password)
        else:
            _creds = self.get_credentials(
                conf, username, tenant_name, password,
                identity_version=self.identity_version)

        _auth = self.get_auth_provider(conf, _creds)
        self.auth_provider = _auth

        if "v2.0" in conf.get("identity", "uri"):
            self.identity = identity_client.IdentityClient(
                _auth, conf.get_defaulted('identity', 'catalog_type'),
                self.identity_region, endpoint_type='adminURL',
                **default_params)
        else:
            self.identity = identity_v3_client.IdentityV3Client(
                _auth, conf.get_defaulted('identity', 'catalog_type'),
                self.identity_region, endpoint_type='adminURL',
                **default_params)

        self.tenants = tenants_client.TenantsClient(
            _auth,
            conf.get_defaulted('identity', 'catalog_type'),
            self.identity_region,
            endpoint_type='adminURL',
            **default_params)

        self.roles = roles_client.RolesClient(
            _auth,
            conf.get_defaulted('identity', 'catalog_type'),
            self.identity_region,
            endpoint_type='adminURL',
            **default_params)

        self.users = users_client.UsersClient(
            _auth,
            conf.get_defaulted('identity', 'catalog_type'),
            self.identity_region,
            endpoint_type='adminURL',
            **default_params)

        self.images = images_client.ImagesClient(
            _auth,
            conf.get_defaulted('image', 'catalog_type'),
            self.identity_region,
            **default_params)
        self.servers = servers_client.ServersClient(_auth,
                                                    **compute_params)
        self.flavors = flavors_client.FlavorsClient(_auth,
                                                    **compute_params)

        self.networks = None

        def create_nova_network_client():
            if self.networks is None:
                self.networks = nova_net_client.NetworksClient(
                    _auth, **compute_params)
            return self.networks

        def create_neutron_client():
            if self.networks is None:
                self.networks = networks_client.NetworksClient(
                    _auth,
                    conf.get_defaulted('network', 'catalog_type'),
                    self.identity_region,
                    endpoint_type=conf.get_defaulted('network',
                                                     'endpoint_type'),
                    **default_params)
            return self.networks

        self.get_nova_net_client = create_nova_network_client
        self.get_neutron_client = create_neutron_client

        # Set admin tenant id needed for keystone v3 tests.
        if admin:
            tenant_id = identity.get_tenant_by_name(self.tenants,
                                                    tenant_name)['id']
            conf.set('identity', 'admin_tenant_id', tenant_id)


def create_tempest_users(tenants_client, roles_client, users_client, conf,
                         services):
    """Create users necessary for Tempest if they don't exist already."""
    create_user_with_tenant(tenants_client, users_client,
                            conf.get('identity', 'username'),
                            conf.get('identity', 'password'),
                            conf.get('identity', 'tenant_name'))

    give_role_to_user(tenants_client, roles_client, users_client,
                      conf.get('identity', 'admin_username'),
                      conf.get('identity', 'tenant_name'), role_name='admin')

    # Prior to juno, and with earlier juno defaults, users needed to have
    # the heat_stack_owner role to use heat stack apis. We assign that role
    # to the user if the role is present.
    if 'orchestration' in services:
        give_role_to_user(tenants_client, roles_client, users_client,
                          conf.get('identity', 'username'),
                          conf.get('identity', 'tenant_name'),
                          role_name='heat_stack_owner',
                          role_required=False)

    create_user_with_tenant(tenants_client, users_client,
                            conf.get('identity', 'alt_username'),
                            conf.get('identity', 'alt_password'),
                            conf.get('identity', 'alt_tenant_name'))


def give_role_to_user(tenants_client, roles_client, users_client, username,
                      tenant_name, role_name, role_required=True):
    """Give the user a role in the project (tenant).""",

    tenant_id = identity.get_tenant_by_name(tenants_client, tenant_name)['id']
    users = users_client.list_users()
    user_ids = [u['id'] for u in users['users'] if u['name'] == username]
    user_id = user_ids[0]
    roles = roles_client.list_roles()
    role_ids = [r['id'] for r in roles['roles'] if r['name'] == role_name]
    if not role_ids:
        if role_required:
            raise Exception("required role %s not found" % role_name)
        LOG.debug("%s role not required" % role_name)
        return
    role_id = role_ids[0]
    try:
        roles_client.create_user_role_on_project(tenant_id, user_id, role_id)
        LOG.debug("User '%s' was given the '%s' role in project '%s'",
                  username, role_name, tenant_name)
    except exceptions.Conflict:
        LOG.debug("(no change) User '%s' already has the '%s' role in"
                  " project '%s'", username, role_name, tenant_name)


def create_user_with_tenant(tenants_client, users_client, username,
                            password, tenant_name):
    """Create user and tenant if he doesn't exist.

    Sets password even for existing user.
    """
    LOG.info("Creating user '%s' with tenant '%s' and password '%s'",
             username, tenant_name, password)
    tenant_description = "Tenant for Tempest %s user" % username
    email = "%s@test.com" % username
    # create tenant
    try:
        tenants_client.create_tenant(name=tenant_name,
                                     description=tenant_description)
    except exceptions.Conflict:
        LOG.info("(no change) Tenant '%s' already exists", tenant_name)

    tenant_id = identity.get_tenant_by_name(tenants_client, tenant_name)['id']
    # create user
    try:
        users_client.create_user(**{'name': username, 'password': password,
                                    'tenantId': tenant_id, 'email': email})
    except exceptions.Conflict:
        LOG.info("User '%s' already exists. Setting password to '%s'",
                 username, password)
        user = identity.get_user_by_username(tenants_client, tenant_id,
                                             username)
        users_client.update_user_password(user['id'], password=password)


def create_tempest_flavors(client, conf, allow_creation):
    """Find or create flavors 'm1.nano' and 'm1.micro' and set them in conf.

    If 'flavor_ref' and 'flavor_ref_alt' are specified in conf, it will first
    try to find those - otherwise it will try finding or creating 'm1.nano' and
    'm1.micro' and overwrite those options in conf.

    :param allow_creation: if False, fail if flavors were not found
    """
    # m1.nano flavor
    flavor_id = None
    if conf.has_option('compute', 'flavor_ref'):
        flavor_id = conf.get('compute', 'flavor_ref')
    flavor_id = find_or_create_flavor(client,
                                      flavor_id, 'm1.nano',
                                      allow_creation, ram=64)
    conf.set('compute', 'flavor_ref', flavor_id)

    # m1.micro flavor
    alt_flavor_id = None
    if conf.has_option('compute', 'flavor_ref_alt'):
        alt_flavor_id = conf.get('compute', 'flavor_ref_alt')
    alt_flavor_id = find_or_create_flavor(client,
                                          alt_flavor_id, 'm1.micro',
                                          allow_creation, ram=128)
    conf.set('compute', 'flavor_ref_alt', alt_flavor_id)


def find_or_create_flavor(client, flavor_id, flavor_name,
                          allow_creation, ram=64, vcpus=1, disk=0):
    """Try finding flavor by ID or name, create if not found.

    :param flavor_id: first try finding the flavor by this
    :param flavor_name: find by this if it was not found by ID, create new
        flavor with this name if not found at all
    :param allow_creation: if False, fail if flavors were not found
    :param ram: memory of created flavor in MB
    :param vcpus: number of VCPUs for the flavor
    :param disk: size of disk for flavor in GB
    """
    flavor = None
    flavors = client.list_flavors()['flavors']
    # try finding it by the ID first
    if flavor_id:
        found = [f for f in flavors if f['id'] == flavor_id]
        if found:
            flavor = found[0]
    # if not found previously, try finding it by name
    if flavor_name and not flavor:
        found = [f for f in flavors if f['name'] == flavor_name]
        if found:
            flavor = found[0]

    if not flavor and not allow_creation:
        raise Exception("Flavor '%s' not found, but resource creation"
                        " isn't allowed. Either use '--create' or provide"
                        " an existing flavor" % flavor_name)

    if not flavor:
        LOG.info("Creating flavor '%s'", flavor_name)
        flavor = client.create_flavor(name=flavor_name,
                                      ram=ram, vcpus=vcpus,
                                      disk=disk, id=None)
        return flavor['flavor']['id']
    else:
        LOG.info("(no change) Found flavor '%s'", flavor['name'])

    return flavor['id']


def create_tempest_images(client, conf, image_path, allow_creation,
                          disk_format):
    img_path = unfrackpath(os.path.join(conf.get("scenario", "img_dir"),
                                        conf.get_defaulted("scenario", "img_file")))

    prepare_path(img_path)  # create path if doesn't exists

    name = image_path[image_path.rfind('/') + 1:]
    alt_name = name + "_alt"
    image_id = None
    if conf.has_option('compute', 'image_ref'):
        image_id = conf.get('compute', 'image_ref')
    image_id = find_or_upload_image(client,
                                    image_id, name, allow_creation,
                                    image_source=image_path,
                                    image_dest=img_path,
                                    disk_format=disk_format)
    alt_image_id = None
    if conf.has_option('compute', 'image_ref_alt'):
        alt_image_id = conf.get('compute', 'image_ref_alt')
    alt_image_id = find_or_upload_image(client,
                                        alt_image_id, alt_name, allow_creation,
                                        image_source=image_path,
                                        image_dest=img_path,
                                        disk_format=disk_format)

    conf.set('compute', 'image_ref', image_id)
    conf.set('compute', 'image_ref_alt', alt_image_id)


def find_or_upload_image(client, image_id, image_name, allow_creation,
                         image_source='', image_dest='', disk_format=''):
    image = _find_image(client, image_id, image_name)
    if not image and not allow_creation:
        raise Exception("Image '%s' not found, but resource creation"
                        " isn't allowed. Either use '--create' or provide"
                        " an existing image_ref" % image_name)

    if image:
        LOG.info("(no change) Found image '%s'", image['name'])
        path = os.path.abspath(image_dest)
        if not os.path.isfile(path):
            _download_image(client, image['id'], path)
    else:
        LOG.info("Creating image '%s'", image_name)
        if image_source.startswith("http:") or \
                image_source.startswith("https:"):
            _download_file(image_source, image_dest)
        else:
            shutil.copyfile(image_source, image_dest)
        image = _upload_image(client, image_name, image_dest, disk_format)
    return image['id']


def create_tempest_networks(clients, conf, has_neutron, public_network_id):
    label = None
    # TODO(tkammer): separate logic to different func of Nova network
    # vs Neutron
    if has_neutron:
        client = clients.get_neutron_client()

        # if user supplied the network we should use
        if public_network_id:
            LOG.info("Looking for existing network id: {0}"
                     "".format(public_network_id))

            # check if network exists
            network_list = client.list_networks()
            for network in network_list['networks']:
                if network['id'] == public_network_id:
                    break
            else:
                raise ValueError('provided network id: {0} was not found.'
                                 ''.format(public_network_id))

        # no network id provided, try to auto discover a public network
        else:
            LOG.info("No network supplied, trying auto discover for network")
            network_list = client.list_networks()
            for network in network_list['networks']:
                if network['router:external'] and network['subnets']:
                    LOG.info("Found network, using: {0}".format(network['id']))
                    public_network_id = network['id']
                    break

            # Couldn't find an existing external network
            else:
                LOG.error("No external networks found. "
                          "Please note that any test that relies on external "
                          "connectivity would most likely fail.")

        if public_network_id is not None:
            conf.set('network', 'public_network_id', public_network_id)

    else:
        client = clients.get_nova_net_client()
        networks = client.list_networks()
        if networks:
            label = networks['networks'][0]['label']

    if label:
        conf.set('compute', 'fixed_network_name', label)
    elif not has_neutron:
        raise Exception('fixed_network_name could not be discovered and'
                        ' must be specified')


def configure_boto(conf, services):
    """Set boto URLs based on discovered APIs."""
    if 'ec2' in services:
        conf.set('boto', 'ec2_url', services['ec2']['url'])
    if 's3' in services:
        conf.set('boto', 's3_url', services['s3']['url'])


def configure_horizon(conf):
    """Derive the horizon URIs from the identity's URI."""
    uri = conf.get('identity', 'uri')
    base = uri.rsplit(':', 1)[0] + '/dashboard'
    assert base.startswith('http:') or base.startswith('https:')
    has_horizon = True
    try:
        ansible.module_utils.urls.open_url(base)
    except Exception:
        has_horizon = False
    conf.set('service_available', 'horizon', str(has_horizon))
    conf.set('dashboard', 'dashboard_url', base + '/')
    conf.set('dashboard', 'login_url', base + '/auth/login/')


def configure_discovered_services(conf, services):
    """Set service availability and supported extensions and versions.

    Set True/False per service in the [service_available] section of `conf`
    depending of wheter it is in services. In the [<service>-feature-enabled]
    section, set extensions and versions found in `services`.

    :param conf: ConfigParser configuration
    :param services: dictionary of discovered services - expects each service
        to have a dictionary containing 'extensions' and 'versions' keys
    """
    # check if volume service is disabled
    if conf.has_section('services') and conf.has_option('services', 'volume'):
        if not conf.getboolean('services', 'volume'):
            SERVICE_NAMES.pop('volume')
            SERVICE_VERSIONS.pop('volume')
    # set service availability
    for service, codename in SERVICE_NAMES.iteritems():
        # ceilometer is still transitioning from metering to telemetry
        if service == 'telemetry' and 'metering' in services:
            service = 'metering'
        # data-processing is the default service name since Kilo
        elif service == 'data-processing' and 'data_processing' in services:
            service = 'data_processing'
        conf.set('service_available', codename, str(service in services))

    # set supported API versions for services with more of them
    for service, versions in SERVICE_VERSIONS.iteritems():
        supported_versions = services.get(service, {}).get('versions', [])
        section = service + '-feature-enabled'
        for version in versions:
            is_supported = any(version in item
                               for item in supported_versions)
            conf.set(section, 'api_' + version, str(is_supported))

    # set service extensions
    keystone_v3_support = conf.get('identity-feature-enabled', 'api_v3')
    for service, ext_key in SERVICE_EXTENSION_KEY.iteritems():
        if service in services:
            extensions = ','.join(services[service]['extensions'])
            if service == 'object-store':
                # tempest.conf is inconsistent and uses 'object-store' for the
                # catalog name but 'object-storage-feature-enabled'
                service = 'object-storage'
            if service == 'identity' and keystone_v3_support:
                identity_v3_ext = get_identity_v3_extensions(
                    conf.get("identity", "uri_v3"))
                extensions = list(set(extensions.split(',') + identity_v3_ext))
                extensions = ','.join(extensions)
            conf.set(service + '-feature-enabled', ext_key, extensions)


def _download_file(url, destination):
    LOG.info("Downloading '%s' and saving as '%s'", url, destination)
    f = ansible.module_utils.urls.open_url(url)  # urllib2.urlopen(url)
    data = f.read()
    with open(destination, "wb") as dest:
        dest.write(data)


def _download_image(client, id, path):
    """Download file from glance."""
    LOG.info("Downloading image %s to %s" % (id, path))
    body = client.show_image_file(id)
    LOG.debug(type(body.data))
    with open(path, 'wb') as out:
        out.write(body.data)


def _upload_image(client, name, path, disk_format):
    """Upload image file from `path` into Glance with `name."""
    LOG.info("Uploading image '%s' from '%s'", name, os.path.abspath(path))

    with open(path) as data:
        image = client.create_image(name=name,
                                    disk_format=disk_format,
                                    container_format='bare',
                                    visibility="public")
        client.store_image_file(image['id'], data)
        return image


def _find_image(client, image_id, image_name):
    """Find image by ID or name (the image client doesn't have this)."""

    if image_id:
        try:
            return client.get_image(image_id)
        except exceptions.NotFound:
            pass
    found = filter(lambda x: x['name'] == image_name,
                   client.list_images()['images'])
    if found:
        return found[0]
    else:
        return None


class TempestConf(ConfigParser.SafeConfigParser):
    def __init__(self, log_file=os.devnull):
        ConfigParser.SafeConfigParser.__init__(self)
        # causes the config parser to preserve case of the options
        self.optionxform = str

        # set of pairs `(section, key)` which have a higher priority (are
        # user-defined) and will usually not be overwritten by `set()`
        self.priority_sectionkeys = set()

        # disable logging TODO find a better way

        tempest.config._CONF.log_dir, tempest.config._CONF.log_file = os.path.split(log_file)
        tempest.config._CONF.use_stderr = False
        self.CONF = tempest.config.TempestConfigPrivate(parse_conf=False)

    def get_bool_value(self, value):
        strval = str(value).lower()
        if strval == 'true':
            return True
        elif strval == 'false':
            return False
        else:
            raise ValueError("'%s' is not a boolean" % value)

    def get_defaulted(self, section, key):
        if self.has_option(section, key):
            return self.get(section, key)
        else:
            return self.CONF.get(section).get(key)

    def set(self, section, key, value, priority=False):
        """Set value in configuration, similar to `SafeConfigParser.set`

        Creates non-existent sections. Keeps track of options which were
        specified by the user and should not be normally overwritten.

        :param priority: if True, always over-write the value. If False, don't
            over-write an existing value if it was written before with a
            priority (i.e. if it was specified by the user)
        :returns: True if the value was written, False if not (because of
            priority)
        """
        if not self.has_section(section) and section.lower() != "default":
            self.add_section(section)
        if not priority and (section, key) in self.priority_sectionkeys:
            LOG.debug("Option '[%s] %s = %s' was defined by user, NOT"
                      " overwriting into value '%s'", section, key,
                      self.get(section, key), value)
            return False
        if priority:
            self.priority_sectionkeys.add((section, key))
        LOG.debug("Setting [%s] %s = %s", section, key, value)
        ConfigParser.SafeConfigParser.set(self, section, key, value)
        return True


# API_DISCOVERY -------------------------------------

MULTIPLE_SLASH = re.compile(r'/+')


class ServiceError(Exception):
    pass


class Service(object):
    def __init__(self, name, service_url, token, disable_ssl_validation):
        self.name = name
        self.service_url = service_url
        self.headers = {'Accept': 'application/json', 'X-Auth-Token': token}
        self.disable_ssl_validation = disable_ssl_validation

    def do_get(self, url, top_level=False, top_level_path=""):

        parts = list(urlparse.urlparse(url))
        # 2 is the path offset
        if top_level:
            parts[2] = '/' + top_level_path

        parts[2] = MULTIPLE_SLASH.sub('/', parts[2])
        url = urlparse.urlunparse(parts)

        try:
            if self.disable_ssl_validation:
                urllib3.disable_warnings()
                http = urllib3.PoolManager(cert_reqs='CERT_NONE')
            else:
                http = urllib3.PoolManager()
            r = http.request('GET', url, headers=self.headers)
        except Exception as e:
            raise Exception("Request on service '%s' with url '%s' failed" %
                            (self.name, url))
        if r.status >= 400:
            raise ServiceError("Request on service '%s' with url '%s' failed"
                               " with code %d" % (self.name, url, r.status))
        return r.data

    def get_extensions(self):
        return []

    def get_versions(self):
        return []


class VersionedService(Service):
    def get_versions(self):
        body = self.do_get(self.service_url, top_level=True)
        body = json.loads(body)
        return self.deserialize_versions(body)

    def deserialize_versions(self, body):
        return map(lambda x: x['id'], body['versions'])


class ComputeService(VersionedService):
    def get_extensions(self):
        body = self.do_get(self.service_url + '/extensions')
        body = json.loads(body)
        return map(lambda x: x['alias'], body['extensions'])


class ImageService(VersionedService):
    pass


class NetworkService(VersionedService):
    def get_extensions(self):
        body = self.do_get(self.service_url + '/v2.0/extensions.json')
        body = json.loads(body)
        return map(lambda x: x['alias'], body['extensions'])


class VolumeService(VersionedService):
    def get_extensions(self):
        body = self.do_get(self.service_url + '/extensions')
        body = json.loads(body)
        return map(lambda x: x['alias'], body['extensions'])


class IdentityService(VersionedService):
    def get_extensions(self):
        if 'v2.0' in self.service_url:
            body = self.do_get(self.service_url + '/extensions')
        else:
            body = self.do_get(self.service_url + '/v2.0/extensions')
        body = json.loads(body)
        return map(lambda x: x['alias'], body['extensions']['values'])

    def deserialize_versions(self, body):
        return map(lambda x: x['id'], body['versions']['values'])


class ObjectStorageService(Service):
    def get_extensions(self):
        body = self.do_get(self.service_url, top_level=True,
                           top_level_path="info")
        body = json.loads(body)
        # Remove Swift general information from extensions list
        body.pop('swift')
        return body.keys()


service_dict = {'compute': ComputeService,
                'image': ImageService,
                'network': NetworkService,
                'object-store': ObjectStorageService,
                'volume': VolumeService,
                'identity': IdentityService}


def get_service_class(service_name):
    return service_dict.get(service_name, Service)


def get_identity_v3_extensions(keystone_v3_url):
    """Returns discovered identity v3 extensions

    As keystone V3 uses a JSON Home to store the extensions,
    this method is kept  here just for the sake of functionality, but it
    implements a different discovery method.

    :param keystone_v3_url: Keystone V3 auth url
    :return: A list with the discovered extensions
    """
    try:
        r = requests.get(keystone_v3_url,
                         verify=False,
                         headers={'Accept': 'application/json-home'})
    except requests.exceptions.RequestException as re:
        raise Exception("Request on service '%s' with url '%s' failed" %
                        ('identity', keystone_v3_url))

    ext_h = 'http://docs.openstack.org/api/openstack-identity/3/ext/'
    res = [x for x in json.loads(r.content)['resources'].keys()]
    ext = [ex for ex in res if 'ext' in ex]
    return list(set([str(e).replace(ext_h, '').split('/')[0] for e in ext]))


def discover(auth_provider, region, object_store_discovery=True,
             api_version=2, disable_ssl_certificate_validation=True):
    """Returns a dict with discovered apis.

    :param auth_provider: An AuthProvider to obtain service urls.
    :param region: A specific region to use. If the catalog has only one region
    then that region will be used.
    :return: A dict with an entry for the type of each discovered service.
        Each entry has keys for 'extensions' and 'versions'.
    """
    token, auth_data = auth_provider.get_auth()
    services = {}
    service_catalog = 'serviceCatalog'
    public_url = 'publicURL'
    identity_port = urlparse.urlparse(auth_provider.auth_url).port
    identity_version = urlparse.urlparse(auth_provider.auth_url).path
    if api_version == 3:
        service_catalog = 'catalog'
        public_url = 'url'

    for entry in auth_data[service_catalog]:
        name = entry['type']
        services[name] = dict()
        for _ep in entry['endpoints']:
            if _ep['region'] == region:
                ep = _ep
                break
        else:
            ep = entry['endpoints'][0]
        if 'identity' in ep[public_url]:
            services[name]['url'] = ep[public_url].replace(
                "/identity", ":{0}{1}".format(
                    identity_port, identity_version))
        else:
            services[name]['url'] = ep[public_url]
        service_class = get_service_class(name)
        service = service_class(name, services[name]['url'], token,
                                disable_ssl_certificate_validation)
        if name == 'object-store' and not object_store_discovery:
            services[name]['extensions'] = []
        else:
            services[name]['extensions'] = service.get_extensions()
        services[name]['versions'] = service.get_versions()
    return services


if __name__ == "__main__":
    main()
